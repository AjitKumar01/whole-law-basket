"""Proposed Whole-Law Basket Models:
1. CompositeWholeLawEnergy: Exact-normalized whole-basket mixture combining
   (a) category-frontier DP energy with nonparametric global count potential g(N),
       shopper count slope s_h*(N-1), and exact size-1 quick-trip utility stratum W_1,
   (b) exact-partition pre-trip category mission mixture (computed via O(P^2) ESP
       without 2^P subset enumeration), and
   (c) exact-partition product bundle mission mixture (computed via O(P^2) ESP).
   Preserves exact input-price autograd derivatives d log P / d dlp.
2. CalibratedMissionLaw: Differentiable O(P*M) exponential-race PL mission law
   supporting arbitrary category size P >= 1, exact Newton count-block polish,
   and validation-selected hierarchical price regularization.
"""

import contextlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .race_pl import race_subset_log_probabilities, race_observed_log_probability


def count_pairs(n: torch.Tensor) -> torch.Tensor:
    n = n.double()
    return n * (n - 1) / 2.0


def log_esp(u: torch.Tensor, max_deg: int | None = None) -> torch.Tensor:
    """All elementary symmetric polynomial log-coefficients up to max_deg in O(P^2) with single allocation."""
    P = u.shape[-1]
    deg = P if max_deg is None else min(P, max_deg)
    coeff = u.new_full((*u.shape[:-1], deg + 1), -torch.inf)
    coeff[..., 0] = 0.0
    need_clone = torch.is_grad_enabled() and u.requires_grad
    for j in range(P):
        hi = min(j + 1, deg)
        prev = coeff[..., :hi].clone() if need_clone else coeff[..., :hi]
        cur = coeff[..., 1 : hi + 1].clone() if need_clone else coeff[..., 1 : hi + 1]
        coeff[..., 1 : hi + 1] = torch.logaddexp(cur, prev + u[..., j : j + 1])
    return coeff


def _sample_esp_subset_local(
    u: torch.Tensor, sizes: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Exact O(P^2) backward-DP subset draw of size `sizes[b]` from items with log-weights `u[b]`."""
    B, P = u.shape
    if P == 0 or B == 0:
        return torch.zeros_like(u)
    suffix = [None] * (P + 1)
    suffix[-1] = u.new_zeros((B, 1))
    max_n = int(sizes.max().item())
    for k in range(P - 1, -1, -1):
        coefficients = suffix[k + 1]
        absent = torch.cat((coefficients, u.new_full((B, 1), -torch.inf)), -1)
        present = torch.cat(
            (u.new_full((B, 1), -torch.inf), coefficients + u[:, k : k + 1]), -1
        )
        nxt = torch.logaddexp(absent, present)
        if nxt.shape[1] > max_n + 1:
            nxt = nxt[:, : max_n + 1]
        suffix[k] = nxt

    remaining = sizes.clone().long()
    row = torch.arange(B, device=u.device)
    output = torch.zeros_like(u)
    for k in range(P):
        n = remaining
        denom = suffix[k][row, n.clamp(max=suffix[k].shape[1] - 1)]
        idx = (n - 1).clamp(min=0, max=suffix[k + 1].shape[1] - 1)
        num = u[:, k] + suffix[k + 1][row, idx]
        prob = (num - denom).exp().clamp(0, 1)
        prob = torch.where(n == 0, 0.0, prob)
        prob = torch.where(n == P - k, 1.0, prob)
        take = torch.rand(B, generator=generator, device=u.device, dtype=u.dtype) < prob
        output[:, k] = take.to(u.dtype)
        remaining -= take.long()
    return output


class StructuredEnergy(nn.Module):
    """Category-frontier dynamic programming (DP) energy model with screened pairwise links."""

    def __init__(self, config: dict, edges: list):
        super().__init__()
        self.config = {
            **config,
            "model_family": "whole_joint_category_count_links",
            "edges": [list(edge) for edge in edges],
        }
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.edges = [tuple(edge) for edge in edges]
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((config["users"], self.J), dtype=torch.float64)
        )
        self.price_below = nn.Parameter(torch.ones(self.J, dtype=torch.float64))
        self.price_above = nn.Parameter(torch.ones(self.J, dtype=torch.float64))
        self.category_count = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.global_count = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.link = nn.Parameter(torch.zeros(len(edges), dtype=torch.float64))
        self.topology, self.interfaces, self.final_counts = self.compile()

    def compile(self):
        endpoints = {j for edge in self.edges for j in edge}
        old_boundary, states = (), [(0, 0)]
        topology, interfaces = [], []
        for c in range(self.C):
            products = tuple(range(c * self.P, (c + 1) * self.P))
            interface = tuple(j for j in products if j in endpoints)
            interfaces.append(interface)
            free = tuple(j for j in products if j not in endpoints)
            new_boundary = tuple(
                sorted(
                    {
                        i
                        for edge in self.edges
                        for i in edge
                        if i // self.P <= c and any(j // self.P > c for j in edge)
                    }
                )
            )
            charged = [
                k
                for k, edge in enumerate(self.edges)
                if max(j // self.P for j in edge) == c
            ]
            records, destinations = [], {}
            for old_index, (old_mask, old_count) in enumerate(states):
                old_bits = {j: (old_mask >> k) & 1 for k, j in enumerate(old_boundary)}
                for mask in range(1 << len(interface)):
                    local_bits = {j: (mask >> k) & 1 for k, j in enumerate(interface)}
                    bits = {**old_bits, **local_bits}
                    selected = sum(local_bits.values())
                    new_mask = sum(bits[j] << k for k, j in enumerate(new_boundary))
                    edge_features = [
                        bits[i] * bits[j] if k in charged else 0
                        for k, (i, j) in enumerate(self.edges)
                    ]
                    for extra in range(len(free) + 1):
                        n = selected + extra
                        key = (new_mask, old_count + n)
                        destination = destinations.setdefault(key, len(destinations))
                        records.append(
                            (old_index, destination, mask, n, extra, edge_features)
                        )
            incoming = [[] for _ in destinations]
            for k, record in enumerate(records):
                incoming[record[1]].append(k)
            max_incoming = max(map(len, incoming))
            padded = np.full((len(incoming), max_incoming), -1, dtype=int)
            for index, paths in enumerate(incoming):
                padded[index, : len(paths)] = paths
            masks = np.array(
                [
                    [(mask >> k) & 1 for k in range(len(interface))]
                    for mask in range(1 << len(interface))
                ],
                dtype=float,
            )
            topology.append(
                {
                    "old": torch.tensor([r[0] for r in records]),
                    "dest": torch.tensor([r[1] for r in records]),
                    "mask": torch.tensor([r[2] for r in records]),
                    "count": torch.tensor([r[3] for r in records]),
                    "extra": torch.tensor([r[4] for r in records]),
                    "edge": torch.tensor(
                        [r[5] for r in records], dtype=torch.float64
                    ).reshape(len(records), len(self.edges)),
                    "incoming": torch.tensor(padded),
                    "states": list(destinations),
                    "bits": torch.tensor(masks, dtype=torch.float64),
                    "interface": interface,
                    "free": free,
                    "boundary": new_boundary,
                }
            )
            old_boundary, states = new_boundary, list(destinations)
        return topology, interfaces, torch.tensor([n for _, n in states])

    def utilities(self, ctx: dict) -> torch.Tensor:
        d = ctx["dlp"].reshape(-1, self.J)
        u = (
            self.lam
            + self.house[ctx["house"]]
            - d.clamp(max=0) * self.price_below
            - d.clamp(min=0) * self.price_above
        )
        if "baseline" in ctx:
            u = ctx["baseline"].reshape(-1, self.J) + u
        return u

    def energy(self, y: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
        counts = y.reshape(-1, self.C, self.P).sum(-1)
        value = (y * utility).sum(-1) + (count_pairs(counts) * self.category_count).sum(
            -1
        )
        value += count_pairs(counts.sum(-1)) * self.global_count[0]
        for k, (i, j) in enumerate(self.edges):
            value = value + self.link[k] * y[:, i] * y[:, j]
        return value

    def partition(
        self,
        utility: torch.Tensor,
        keep_paths: bool = False,
        return_final: bool = False,
    ):
        weights = utility.new_zeros((len(utility), 1))
        layers, paths = [weights], []
        for c, topology in enumerate(self.topology):
            coefficients = log_esp(utility[:, topology["free"]])
            selected_utility = utility[:, topology["interface"]] @ topology["bits"].T
            values = (
                weights[:, topology["old"]]
                + selected_utility[:, topology["mask"]]
                + coefficients[:, topology["extra"]]
                + self.category_count[c] * count_pairs(topology["count"])
                + (topology["edge"] @ self.link)[None]
            )
            destinations = topology["dest"][None].expand(len(utility), -1)
            maximum = values.new_full(
                (len(utility), len(topology["states"])), -torch.inf
            )
            maximum.scatter_reduce_(
                1, destinations, values.detach(), reduce="amax", include_self=True
            )
            mass = torch.zeros_like(maximum)
            mass.scatter_add_(
                1, destinations, (values - maximum.gather(1, destinations)).exp()
            )
            weights = maximum + mass.log()
            if keep_paths:
                layers.append(weights)
                paths.append(values)
        final = weights + self.global_count[0] * count_pairs(self.final_counts)[None]
        final = final.masked_fill(self.final_counts[None] == 0, -torch.inf)
        if return_final and not keep_paths:
            return final
        z = torch.logsumexp(final, -1)
        return (z, layers, paths, final) if keep_paths else z

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        utility = self.utilities(ctx)
        y = y.reshape(-1, self.J).double()
        return self.energy(y, utility) - self.partition(utility)

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        utility = self.utilities(ctx)
        _, _, paths, final = self.partition(utility, keep_paths=True)
        generator = torch.Generator().manual_seed(seed)
        state = torch.multinomial(final.softmax(-1), 1, generator=generator).ravel()
        row = torch.arange(len(utility))
        output = torch.zeros_like(utility)
        for c in range(self.C - 1, -1, -1):
            topology = self.topology[c]
            candidates = topology["incoming"][state]
            candidate_weights = paths[c].gather(1, candidates.clamp(min=0))
            candidate_weights = candidate_weights.masked_fill(
                candidates < 0, -torch.inf
            )
            selected = torch.multinomial(
                candidate_weights.softmax(-1), 1, generator=generator
            ).ravel()
            transition = candidates[row, selected]
            mask = topology["mask"][transition]
            output[:, topology["interface"]] = topology["bits"][mask]
            remaining = topology["extra"][transition].clone()
            free = topology["free"]
            if len(free) > 0:
                output[:, free] = _sample_esp_subset_local(
                    utility[:, free], remaining, generator
                )
            state = topology["old"][transition]
        assert output.any(-1).all()
        return output.reshape(-1, self.C, self.P)


class CompositeWholeLawEnergy(nn.Module):
    """Exact normalized mixture of nonparametric-count DP energy and pre-trip category/product missions."""

    def __init__(
        self,
        config: dict,
        re_model: StructuredEnergy,
        M_cat_init: torch.Tensor,
        M_prod_fixed: torch.Tensor,
        freeze_house_1: bool = False,
        allow_price_mix: bool = False,
    ):
        super().__init__()
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.users = int(config["users"])
        self.allow_price_mix = allow_price_mix
        self.config = {
            **config,
            "model_family": "composite_whole_law_nonparametric_count_and_missions",
            "category_missions": int(len(M_cat_init)),
            "product_missions": int(len(M_prod_fixed)),
            "allow_price_mix": allow_price_mix,
        }
        self.re_model = re_model
        for p in self.re_model.parameters():
            p.requires_grad_(False)

        self.K_c = len(M_cat_init)
        self.K_p = len(M_prod_fixed)
        self.K_tot = 1 + self.K_c + self.K_p
        self.register_buffer("M_prod_fixed", M_prod_fixed.clone().double())

        self.cat_topology = self._compile_within_category()

        max_obs_n = min(self.J, 17)
        self.max_obs_n = max_obs_n
        self.dg = nn.Parameter(torch.zeros(max_obs_n, dtype=torch.float64))
        self.tail_slope = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
        self.s_house = nn.Parameter(torch.zeros(self.users, dtype=torch.float64))
        self.delta_1 = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house_1 = nn.Parameter(
            torch.zeros((self.users, self.J), dtype=torch.float64)
        )
        if freeze_house_1:
            self.house_1.requires_grad_(False)
        self.cross_1_L = nn.Parameter(torch.zeros((self.J, 8), dtype=torch.float64))
        g_gen = torch.Generator().manual_seed(942981)
        self.cross_1_R = nn.Parameter(
            0.02 * torch.randn((self.J, 8), generator=g_gen, dtype=torch.float64)
        )
        self.M_cat = nn.Parameter(M_cat_init.clone().double())
        self.M_prod_cat = nn.Parameter(
            torch.zeros((self.K_p, self.C), dtype=torch.float64)
        )
        self.mix_logits = nn.Parameter(torch.zeros(self.K_tot, dtype=torch.float64))
        self.house_mix = nn.Parameter(
            torch.zeros((self.users, self.K_tot), dtype=torch.float64)
        )
        self.price_mix = nn.Parameter(
            torch.zeros((self.J, self.K_tot), dtype=torch.float64)
        )
        if not allow_price_mix:
            self.price_mix.requires_grad_(False)
        self.cat_shift = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.house_cat = nn.Parameter(
            torch.zeros((self.users, self.C), dtype=torch.float64)
        )
        self.price_cat = nn.Parameter(
            torch.zeros((self.J, self.C), dtype=torch.float64)
        )

    def _compile_within_category(self) -> list[dict]:
        """Compile polynomial O(P^2) within-category ESP transitions for each category c."""
        cat_top = []
        for c in range(self.C):
            within_edges = [
                (k, i % self.P, j % self.P)
                for k, (i, j) in enumerate(self.re_model.edges)
                if i // self.P == c and j // self.P == c
            ]
            interface = tuple(sorted({p for _, a, b in within_edges for p in (a, b)}))
            free = tuple(p for p in range(self.P) if p not in interface)
            n_int = len(interface)
            masks = np.array(
                [[(m >> k) & 1 for k in range(n_int)] for m in range(1 << n_int)],
                dtype=float,
            )
            records = []
            for m in range(1 << n_int):
                local_bits = {interface[k]: (m >> k) & 1 for k in range(n_int)}
                sel_int = sum(local_bits.values())
                edge_val = sum(
                    float(self.re_model.link[k].detach())
                    * local_bits[a]
                    * local_bits[b]
                    for k, a, b in within_edges
                )
                for extra in range(len(free) + 1):
                    n = sel_int + extra
                    if n >= 1:
                        quad_val = float(self.re_model.category_count[c].detach()) * (
                            n * (n - 1.0) / 2.0
                        )
                        records.append((m, extra, n, edge_val + quad_val))
            cat_top.append(
                {
                    "interface": interface,
                    "free": free,
                    "bits": torch.tensor(masks, dtype=torch.float64),
                    "mask_idx": torch.tensor([r[0] for r in records], dtype=torch.long),
                    "extra_idx": torch.tensor(
                        [r[1] for r in records], dtype=torch.long
                    ),
                    "const_bonus": torch.tensor(
                        [r[3] for r in records], dtype=torch.float64
                    ),
                }
            )
        return cat_top

    def _category_transition_logits(self, u_cp: torch.Tensor) -> list[torch.Tensor]:
        lead_shape = u_cp.shape[:-2]
        u_flat = u_cp.reshape(-1, self.C, self.P)
        free_cats = [
            c for c, t in enumerate(self.cat_topology) if len(t["interface"]) == 0
        ]
        out = [None] * self.C
        if len(free_cats) > 0:
            esp_free = log_esp(u_flat[:, free_cats, :])
            n = torch.arange(1, self.P + 1, dtype=torch.float64, device=u_cp.device)
            bonus = (
                self.re_model.category_count[free_cats, None]
                * (n * (n - 1.0) / 2.0)[None, :]
            )
            trans_free = esp_free[..., 1:] + bonus[None, :, :]
            for idx, c in enumerate(free_cats):
                out[c] = trans_free[:, idx, :].reshape(*lead_shape, self.P)

        for c, top in enumerate(self.cat_topology):
            if out[c] is not None:
                continue
            u_c = u_flat[:, c, :]
            coeff = log_esp(u_c[:, top["free"]])
            sel_u = (
                u_c[:, top["interface"]] @ top["bits"].T
                if len(top["interface"]) > 0
                else u_c.new_zeros((len(u_c), 1))
            )
            vals = (
                sel_u[:, top["mask_idx"]]
                + coeff[:, top["extra_idx"]]
                + top["const_bonus"][None, :]
            )
            out[c] = vals.reshape(*lead_shape, -1)
        return out

    def _category_inclusive_weights(self, u_cp: torch.Tensor) -> torch.Tensor:
        if self.P <= 8:
            if not hasattr(self, "_fast_sub_ne"):
                subs = torch.tensor(
                    [
                        [int((m >> j) & 1) for j in range(self.P)]
                        for m in range(1, 1 << self.P)
                    ],
                    dtype=torch.float64,
                    device=u_cp.device,
                )
                cat_edge = torch.zeros(
                    (self.C, len(subs)), dtype=torch.float64, device=u_cp.device
                )
                for k, (i, j) in enumerate(self.re_model.edges):
                    if i // self.P == j // self.P:
                        c = i // self.P
                        cat_edge[c] += float(self.re_model.link[k].detach()) * (
                            subs[:, i % self.P] * subs[:, j % self.P]
                        )
                n_pairs = subs.sum(-1) * (subs.sum(-1) - 1.0) / 2.0
                base_h = (
                    self.re_model.category_count.detach()[:, None] * n_pairs[None, :]
                    + cat_edge
                )
                self._fast_sub_ne = (subs, base_h)
            subs, base_h = self._fast_sub_ne
            return torch.logsumexp(u_cp @ subs.T + base_h, dim=-1)
        trans_list = self._category_transition_logits(u_cp)
        return torch.stack([torch.logsumexp(t, dim=-1) for t in trans_list], dim=-1)

    def build_g(self):
        g = torch.full((self.J + 1,), -torch.inf, dtype=torch.float64)
        g[1] = 0.0
        g[2 : self.max_obs_n + 1] = self.dg[1:]
        if self.J > self.max_obs_n:
            tail_len = self.J - self.max_obs_n
            g[self.max_obs_n + 1 :] = self.dg[-1] - F.softplus(
                self.tail_slope
            ) * torch.arange(1, tail_len + 1, dtype=torch.float64)
        return g

    def within_category_energy(
        self, y_flat: torch.Tensor, u_base: torch.Tensor
    ) -> torch.Tensor:
        counts = y_flat.reshape(-1, self.C, self.P).sum(-1)
        val = (y_flat * u_base).sum(-1) + (
            count_pairs(counts) * self.re_model.category_count
        ).sum(-1)
        for k, (i, j) in enumerate(self.re_model.edges):
            if i // self.P == j // self.P:
                val = val + self.re_model.link[k] * y_flat[:, i] * y_flat[:, j]
        return val

    def precompute_split(self, ctx, y=None):
        grad_needed = torch.is_grad_enabled() and any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        ctx_mgr = contextlib.nullcontext() if grad_needed else torch.no_grad()
        with ctx_mgr:
            u_base = self.re_model.utilities(ctx)
            parts = []
            for i in range(0, len(u_base), 8192):
                parts.append(
                    self.re_model.partition(u_base[i : i + 8192], return_final=True)
                )
            final_base = torch.cat(parts, 0)
            fc = self.re_model.final_counts
            quad = self.re_model.global_count[0] * count_pairs(fc)
            final_unquad = final_base - quad[None, :]
            cols_by_N = [
                torch.full(
                    (len(u_base),),
                    -torch.inf,
                    dtype=torch.float64,
                    device=u_base.device,
                )
            ]
            for n in range(1, self.J + 1):
                idx = (fc == n).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    cols_by_N.append(
                        torch.full(
                            (len(u_base),),
                            -torch.inf,
                            dtype=torch.float64,
                            device=u_base.device,
                        )
                    )
                elif len(idx) == 1:
                    cols_by_N.append(final_unquad[:, idx[0]])
                else:
                    cols_by_N.append(torch.logsumexp(final_unquad[:, idx], dim=-1))
            W_base = torch.stack(cols_by_N, dim=-1)

            u_cp = u_base.reshape(-1, self.C, self.P)
            log_W_cat_base = self._category_inclusive_weights(u_cp)

            m_prod_cp = self.M_prod_fixed.reshape(self.K_p, self.C, self.P)
            prod_parts = []
            for i in range(0, len(u_base), 1024):
                u_shifted = u_cp[i : i + 1024, None, :, :] + m_prod_cp[None, :, :, :]
                prod_parts.append(self._category_inclusive_weights(u_shifted))
            log_W_prod_base = torch.cat(prod_parts, 0)

            cache = {
                "u_base": u_base,
                "W_base": W_base,
                "log_W_cat_base": log_W_cat_base,
                "log_W_prod_base": log_W_prod_base,
            }
            if y is not None:
                y_flat = y.reshape(-1, self.J).double()
                N_obs = y_flat.sum(-1).long()
                e_base = self.re_model.energy(
                    y_flat, u_base
                ) - self.re_model.global_count[0] * (
                    N_obs.double() * (N_obs.double() - 1.0) / 2.0
                )
                e_within = self.within_category_energy(y_flat, u_base)
                pres = (y.reshape(-1, self.C, self.P).sum(-1) > 0).double()
                y_m_prod = y_flat @ self.M_prod_fixed.T
                cache.update(
                    {
                        "y_flat": y_flat,
                        "N_obs": N_obs,
                        "e_base": e_base,
                        "e_within": e_within,
                        "present": pres,
                        "y_m_prod": y_m_prod,
                    }
                )
                mission_basis_trainable = any(
                    p.requires_grad
                    for p in (
                        self.M_cat,
                        self.M_prod_cat,
                        self.cat_shift,
                        self.house_cat,
                        self.price_cat,
                    )
                )
                if not grad_needed and not mission_basis_trainable:
                    H = ctx["house"]
                    d = ctx["dlp"].reshape(len(H), self.J)
                    d_cat = self.cat_shift + self.house_cat[H] + d @ self.price_cat
                    obs_d_cat = (pres * d_cat).sum(-1)
                    base_m_e = (e_within + obs_d_cat)[:, None]
                    log_W_c = (log_W_cat_base + d_cat)[:, None, :] + self.M_cat[
                        None, :, :
                    ]
                    log_all_c = F.softplus(log_W_c).sum(dim=-1)
                    log_Z_c = log_all_c + torch.log(
                        (-torch.expm1(-log_all_c)).clamp(min=1e-30)
                    )
                    ll_cat = base_m_e + (pres @ self.M_cat.T) - log_Z_c
                    log_W_p = (
                        log_W_prod_base
                        + d_cat[:, None, :]
                        + self.M_prod_cat[None, :, :]
                    )
                    log_all_p = F.softplus(log_W_p).sum(dim=-1)
                    log_Z_p = log_all_p + torch.log(
                        (-torch.expm1(-log_all_p)).clamp(min=1e-30)
                    )
                    ll_prod = base_m_e + y_m_prod + (pres @ self.M_prod_cat.T) - log_Z_p
                    cache["ll_missions"] = torch.cat([ll_cat, ll_prod], dim=-1)
            return cache

    def forward_cached(self, ctx, cache):
        B = len(ctx["house"])
        H = ctx["house"]
        d = ctx["dlp"].reshape(B, self.J)
        u_base = cache["u_base"]
        W_base = cache["W_base"]
        y_flat = cache["y_flat"]
        N_obs = cache["N_obs"]
        e_base = cache["e_base"]

        counts = torch.arange(self.J + 1, dtype=torch.float64)
        n_minus_1 = (counts - 1.0).clamp(min=0.0)
        g = self.build_g()
        dx_1 = self.delta_1
        if self.house_1.requires_grad or bool(self.house_1.abs().max() > 0):
            dx_1 = dx_1 + self.house_1[H]
        if (
            self.cross_1_L.requires_grad
            or self.cross_1_R.requires_grad
            or bool(self.cross_1_L.abs().max() > 0)
        ):
            dx_1 = (
                dx_1
                + (d @ self.cross_1_R) @ self.cross_1_L.T
                - d * (self.cross_1_L * self.cross_1_R).sum(-1)
            )
        u_1 = u_base + dx_1
        W_1 = torch.logsumexp(u_1, dim=-1, keepdim=True)
        W_mod = torch.cat([W_base[:, :1], W_1, W_base[:, 2:]], dim=-1)
        tilt = g[None, :] + self.s_house[H][:, None] * n_minus_1[None, :]
        logits_0 = W_mod + tilt
        is_1 = N_obs == 1
        e_1 = (u_1 * y_flat).sum(-1)
        e_0 = torch.where(
            is_1, e_1, e_base + g[N_obs] + self.s_house[H] * (N_obs.double() - 1.0)
        )
        ll_0 = e_0 - torch.logsumexp(logits_0, dim=-1)

        mission_basis_trainable = any(
            p.requires_grad
            for p in (
                self.M_cat,
                self.M_prod_cat,
                self.cat_shift,
                self.house_cat,
                self.price_cat,
            )
        )
        if "ll_missions" in cache and not mission_basis_trainable:
            all_comp_ll = torch.cat([ll_0[:, None], cache["ll_missions"]], dim=-1)
        else:
            log_W_cat_base = cache["log_W_cat_base"]
            log_W_prod_base = cache["log_W_prod_base"]
            e_within = cache["e_within"]
            present = cache["present"]
            y_m_prod = cache["y_m_prod"]
            d_cat = self.cat_shift + self.house_cat[H] + d @ self.price_cat
            obs_d_cat = (present * d_cat).sum(-1)
            base_m_e = (e_within + obs_d_cat)[:, None]

            log_W_c = (log_W_cat_base + d_cat)[:, None, :] + self.M_cat[None, :, :]
            log_all_c = F.softplus(log_W_c).sum(dim=-1)
            log_Z_c = log_all_c + torch.log((-torch.expm1(-log_all_c)).clamp(min=1e-30))
            ll_cat = base_m_e + (present @ self.M_cat.T) - log_Z_c

            log_W_p = log_W_prod_base + d_cat[:, None, :] + self.M_prod_cat[None, :, :]
            log_all_p = F.softplus(log_W_p).sum(dim=-1)
            log_Z_p = log_all_p + torch.log((-torch.expm1(-log_all_p)).clamp(min=1e-30))
            ll_prod = base_m_e + y_m_prod + (present @ self.M_prod_cat.T) - log_Z_p
            all_comp_ll = torch.cat([ll_0[:, None], ll_cat, ll_prod], dim=-1)

        pi_logits = self.mix_logits[None, :] + self.house_mix[H]
        if self.allow_price_mix:
            pi_logits = pi_logits + d @ self.price_mix
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return torch.logsumexp(log_pi + all_comp_ll, dim=-1)

    def log_probability(self, ctx, y):
        cache = self.precompute_split(ctx, y)
        return self.forward_cached(ctx, cache)

    @torch.no_grad()
    def sample(self, ctx, seed=0):
        B = len(ctx["house"])
        H = ctx["house"]
        d = ctx["dlp"].reshape(B, self.J)
        if not hasattr(self, "_sample_cache"):
            self._sample_cache = {}
        ck = (id(ctx["dlp"]), B)
        if ck not in self._sample_cache:
            u_b = self.re_model.utilities(ctx)
            _, _, p_b, f_b = self.re_model.partition(u_b, keep_paths=True)
            self._sample_cache = {ck: (u_b, p_b, f_b)}
        u_base, paths, final_base = self._sample_cache[ck]
        fc = self.re_model.final_counts
        quad = self.re_model.global_count[0].detach() * count_pairs(fc)
        final_unquad = final_base - quad[None, :]
        cols_by_N = [
            torch.full((B,), -torch.inf, dtype=torch.float64, device=u_base.device)
        ]
        for n in range(1, self.J + 1):
            idx = (fc == n).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                cols_by_N.append(
                    torch.full(
                        (B,), -torch.inf, dtype=torch.float64, device=u_base.device
                    )
                )
            elif len(idx) == 1:
                cols_by_N.append(final_unquad[:, idx[0]])
            else:
                cols_by_N.append(torch.logsumexp(final_unquad[:, idx], dim=-1))
        W_base = torch.stack(cols_by_N, dim=-1)
        counts = torch.arange(self.J + 1, dtype=torch.float64)
        n_minus_1 = (counts - 1.0).clamp(min=0.0)

        pi_logits = self.mix_logits[None, :] + self.house_mix[H]
        if self.allow_price_mix:
            pi_logits = pi_logits + d @ self.price_mix
        pi_probs = pi_logits.softmax(dim=-1)
        gen = torch.Generator().manual_seed(seed)
        comp = torch.multinomial(pi_probs, 1, generator=gen).ravel()

        output = torch.zeros((B, self.C, self.P), dtype=torch.float64)

        idx0 = torch.where(comp == 0)[0]
        if len(idx0) > 0:
            g = self.build_g()
            H0, d0, u0 = H[idx0], d[idx0], u_base[idx0]
            dx_1 = (
                self.delta_1
                + self.house_1[H0]
                + (d0 @ self.cross_1_R) @ self.cross_1_L.T
                - d0 * (self.cross_1_L * self.cross_1_R).sum(-1)
            )
            u_1 = u0 + dx_1
            W_1 = torch.logsumexp(u_1, dim=-1, keepdim=True)
            W_mod = torch.cat([W_base[idx0, :1], W_1, W_base[idx0, 2:]], dim=-1)
            logits_0 = (
                W_mod + g[None, :] + self.s_house[H0][:, None] * n_minus_1[None, :]
            )
            chosen_n = torch.multinomial(logits_0.softmax(-1), 1, generator=gen).ravel()

            is_1 = chosen_n == 1
            if is_1.any():
                rows_1 = idx0[is_1]
                j_drawn = torch.multinomial(
                    u_1[is_1].softmax(-1), 1, generator=gen
                ).ravel()
                y_1 = torch.zeros((len(rows_1), self.J), dtype=torch.float64)
                y_1[torch.arange(len(rows_1)), j_drawn] = 1.0
                output[rows_1] = y_1.reshape(-1, self.C, self.P)

            is_gt1 = ~is_1
            if is_gt1.any():
                rows_gt = idx0[is_gt1]
                req_n = chosen_n[is_gt1]
                f_rows = final_unquad[rows_gt].clone()
                mask_n = fc[None, :] == req_n[:, None]
                f_rows = f_rows.masked_fill(~mask_n, -torch.inf)
                state = torch.multinomial(f_rows.softmax(-1), 1, generator=gen).ravel()
                u_gt = u_base[rows_gt]
                row_arange = torch.arange(len(rows_gt))
                y_gt = torch.zeros_like(u_gt)
                for c in range(self.C - 1, -1, -1):
                    topology = self.re_model.topology[c]
                    candidates = topology["incoming"][state]
                    cand_w = paths[c][rows_gt].gather(1, candidates.clamp(min=0))
                    cand_w = cand_w.masked_fill(candidates < 0, -torch.inf)
                    sel = torch.multinomial(
                        cand_w.softmax(-1), 1, generator=gen
                    ).ravel()
                    trans = candidates[row_arange, sel]
                    mask = topology["mask"][trans]
                    y_gt[:, topology["interface"]] = topology["bits"][mask]
                    rem = topology["extra"][trans].clone()
                    free = topology["free"]
                    if len(free) > 0:
                        y_gt[:, free] = _sample_esp_subset_local(
                            u_gt[:, free], rem, gen
                        )
                    state = topology["old"][trans]
                output[rows_gt] = y_gt.reshape(-1, self.C, self.P)

        idx_m = torch.where(comp >= 1)[0]
        if len(idx_m) > 0:
            Hm, dm, um = H[idx_m], d[idx_m], u_base[idx_m]
            cm = comp[idx_m]
            d_cat = self.cat_shift + self.house_cat[Hm] + dm @ self.price_cat
            is_prod_m = cm > self.K_c
            m_shift = torch.zeros_like(um)
            cat_shift_m = torch.zeros((len(idx_m), self.C), dtype=torch.float64)
            if (~is_prod_m).any():
                kc = cm[~is_prod_m] - 1
                cat_shift_m[~is_prod_m] = self.M_cat[kc]
            if is_prod_m.any():
                kp = cm[is_prod_m] - 1 - self.K_c
                m_shift[is_prod_m] = self.M_prod_fixed[kp]
                cat_shift_m[is_prod_m] = self.M_prod_cat[kp]

            u_eff = (um + m_shift).reshape(-1, self.C, self.P)
            trans_list = self._category_transition_logits(u_eff)
            log_W_m = (
                torch.stack([torch.logsumexp(t, dim=-1) for t in trans_list], dim=-1)
                + d_cat
                + cat_shift_m
            )

            eta = log_W_m
            suffix = [torch.full((len(idx_m),), -torch.inf, dtype=torch.float64)]
            for c in range(self.C - 1, -1, -1):
                suffix.append(
                    torch.logaddexp(
                        F.logsigmoid(eta[:, c]), F.logsigmoid(-eta[:, c]) + suffix[-1]
                    )
                )
            suffix.reverse()
            active = torch.zeros(len(idx_m), dtype=torch.bool)
            pres_draw = torch.zeros((len(idx_m), self.C), dtype=torch.bool)
            for c in range(self.C):
                p_c = torch.where(
                    active,
                    eta[:, c].sigmoid(),
                    (F.logsigmoid(eta[:, c]) - suffix[c]).exp(),
                ).clamp(0, 1)
                take_c = (
                    torch.rand(len(idx_m), generator=gen, dtype=torch.float64) < p_c
                )
                pres_draw[:, c] = take_c
                active |= take_c

            out_m = torch.zeros((len(idx_m), self.C, self.P), dtype=torch.float64)
            for c, top in enumerate(self.cat_topology):
                act_rows = pres_draw[:, c].nonzero(as_tuple=True)[0]
                if len(act_rows) == 0:
                    continue
                t_probs = trans_list[c][act_rows].softmax(dim=-1)
                chosen_rec = torch.multinomial(t_probs, 1, generator=gen).ravel()
                if len(top["interface"]) > 0:
                    m_idx = top["mask_idx"][chosen_rec]
                    out_m[act_rows[:, None], c, list(top["interface"])] = top["bits"][
                        m_idx
                    ]
                rem_extra = top["extra_idx"][chosen_rec]
                if len(top["free"]) > 0:
                    u_free = u_eff[act_rows, c][:, list(top["free"])]
                    out_m[act_rows[:, None], c, list(top["free"])] = (
                        _sample_esp_subset_local(u_free, rem_extra, gen)
                    )
            output[idx_m] = out_m

        assert output.any(dim=(1, 2)).all(), "Nonempty conditioning failed"
        return output


class CalibratedMissionLaw(nn.Module):
    """Joint pre-trip mission + count + differentiable O(P*M) exponential-race PL law."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = {
            **config,
            "model_family": "joint_pretrip_mission_count_race_PL",
            "mission_components": 2,
        }
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P

        def parameter(shape, value=0.0):
            return nn.Parameter(torch.full(shape, value, dtype=torch.float64))

        self.lam = parameter((self.C, self.P))
        self.house = parameter((config["users"], self.C, self.P))
        self.price = parameter((self.C, self.P), 1.0)
        self.product_contrast = parameter((self.C, self.P))
        self.need = parameter((self.C,))
        self.house_need = parameter((config["users"], self.C))
        self.need_contrast = parameter((self.C,))
        self.category_price = parameter((self.C,), 1.0)
        self.category_cross = parameter((self.C,))
        self.category_hinge = parameter((self.C,))
        self.count_need = parameter((self.C,))
        self.count_contrast = parameter((self.C,))
        self.count_price = parameter((self.C,), 1.0)
        self.count_offsets = parameter((self.C, max(self.P - 2, 0)))
        self.mixture_logit = parameter((1,))

    @torch.no_grad()
    def initialize(self, train: tuple[dict, torch.Tensor]):
        ctx, y = train
        y = y.double()
        presence = (y.sum(-1) > 0).double()
        global_incidence = (y.sum(0) + 0.5) / (len(y) + 1.0)
        self.lam.copy_(
            global_incidence.log() - global_incidence.log().mean(-1, keepdim=True)
        )
        observed = presence.mean(0)
        self.need.copy_(torch.logit((0.8 * observed).clamp(0.05, 0.95)))
        for h in range(int(self.config["users"])):
            rows = ctx["house"] == h
            if not rows.any():
                continue
            frequency = (y[rows].sum(0) + 20.0 * global_incidence) / (
                int(rows.sum()) + 20.0
            )
            offset = frequency.log() - global_incidence.log()
            self.house[h].copy_(offset - offset.mean(-1, keepdim=True))
            need = (presence[rows].sum(0) + 20.0 * observed) / (int(rows.sum()) + 20.0)
            self.house_need[h].copy_(
                torch.logit(need.clamp(0.05, 0.95))
                - torch.logit(observed.clamp(0.05, 0.95))
            )
        if self.C > 1:
            covariance = torch.cov(presence.T).numpy()
            _, vectors = np.linalg.eigh(covariance)
            contrast = vectors[:, -1]
            contrast *= 1.0 if contrast[0] >= 0 else -1.0
            self.need_contrast.copy_(torch.tensor(1.5 * contrast, dtype=torch.float64))
        n = y.sum(-1).long()
        for c in range(self.C):
            counts = torch.bincount(n[:, c], minlength=self.P + 1)[1:].double() + 0.5
            if self.P >= 2:
                eta = float((counts[1] / counts[0]).log())
                self.count_need[c] = eta
                for k in range(2, self.P):
                    if not (n[:, c] == k + 1).any():
                        self.count_offsets[c, k - 2] = -30.0
                    else:
                        self.count_offsets[c, k - 2] = (
                            counts[k] / counts[0]
                        ).log() - k * eta

    def parameter_bounds(self) -> list[tuple[float, float]]:
        positive = {"price", "category_price", "category_hinge", "count_price"}
        return [
            (
                (0.0, 8.0)
                if name in positive
                else (-30.0, 15.0) if name == "count_offsets" else (-15.0, 15.0)
            )
            for name, p in self.named_parameters()
            for _ in range(p.numel())
        ]

    def _build_count_logits(self, count_eta: torch.Tensor) -> torch.Tensor:
        terms = [torch.zeros_like(count_eta)]
        if self.P >= 2:
            terms.append(count_eta)
        for k in range(2, self.P):
            terms.append(k * count_eta + self.count_offsets[:, k - 2])
        return torch.stack(terms, dim=-1)

    def parts(self, ctx: dict, include_subset_table: bool = False):
        d = ctx["dlp"]
        index = d.mean(-1)
        signs = d.new_tensor([0.5, -0.5])[None, :, None]
        eta = (
            self.need
            + self.house_need[ctx["house"]]
            - self.category_price * index
            + self.category_cross * index.roll(1, -1)
            - self.category_hinge * index.clamp(min=0)
        )
        eta = eta[:, None] + signs * self.need_contrast
        count_eta = (self.count_need - self.count_price * index)[
            :, None
        ] + signs * self.count_contrast
        count_logits = self._build_count_logits(count_eta)
        count = F.log_softmax(count_logits, -1)
        u = (self.lam + self.house[ctx["house"]] - self.price * d)[:, None]
        u = u + signs[..., None] * self.product_contrast
        subset = (
            race_subset_log_probabilities(u, n_nodes=128)
            if include_subset_table
            else None
        )
        empty = F.logsigmoid(-eta).sum(-1)
        log_nonempty = torch.log((-torch.expm1(empty)).clamp(min=1e-30))
        log_mixture = torch.stack(
            (F.logsigmoid(self.mixture_logit[0]), F.logsigmoid(-self.mixture_logit[0]))
        )
        return eta, count, subset, log_nonempty, log_mixture, u

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        eta, count, _, normalizer, mixture, u = self.parts(
            ctx, include_subset_table=False
        )
        y_d = y.double()
        n = y_d.sum(-1).long()
        present = n > 0
        if (n > self.P).any():
            return torch.full(
                (len(y_d),), -torch.inf, dtype=torch.float64, device=y.device
            )
        choice = race_observed_log_probability(
            u, y_d[:, None, :, :].expand(-1, 2, -1, -1)
        )
        size = count.gather(
            -1,
            (n - 1).clamp(min=0, max=self.P - 1)[:, None, :, None].expand(-1, 2, -1, 1),
        ).squeeze(-1)
        participation = torch.where(
            present[:, None], F.logsigmoid(eta), F.logsigmoid(-eta)
        )
        component = (
            participation + torch.where(present[:, None], size + choice, 0.0)
        ).sum(-1) - normalizer
        result = torch.logsumexp(component + mixture, -1)
        return result.masked_fill(n.sum(-1) == 0, -torch.inf)

    def forward(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        return self.log_probability(ctx, y)

    def set_regularization_config(self, reg_cfg: dict | None = None) -> None:
        self._reg_config = dict(reg_cfg) if reg_cfg is not None else None

    def regularization_loss(self, moments_d: dict) -> torch.Tensor:
        """Strictly permutation-invariant hierarchical price penalty optimized jointly inside L-BFGS-B.

        Every term is symmetric across categories c in {0, ..., C-1} and products p in {0, ..., P-1}
        (using only cross-category sample means and within-category sample means, with zero dependence
        on category index ordering).
        """
        cfg_reg = getattr(self, "_reg_config", None)
        idx_m = moments_d["category_price"]
        hinge_m = moments_d["category_hinge"]
        cross_m = moments_d["category_cross"]
        price_m = moments_d["price"]

        cp = self.category_price
        ch = self.category_hinge
        cnt = self.count_price
        cr = self.category_cross
        pr = self.price

        if not cfg_reg:
            return (
                (pr.square() * price_m).sum()
                + (cp.square() * idx_m).sum()
                + (cnt.square() * idx_m).sum()
                + (ch.square() * hinge_m).sum()
                + (cr.square() * cross_m).sum()
            )

        m_bar = idx_m.mean()
        p_bar = price_m.mean()
        pr_cm = pr.mean(dim=-1, keepdim=True)

        pen_hinge_var = (
            float(cfg_reg.get("w_hinge_var", 0.0))
            * m_bar
            * (ch - ch.mean()).square().sum()
        )
        pen_cross_var = (
            float(cfg_reg.get("w_cross_var", 0.0))
            * m_bar
            * (cr - cr.mean()).square().sum()
        )
        pen_cat_var = (
            float(cfg_reg.get("w_cat_var", 0.0))
            * m_bar
            * ((cp - cp.mean()).square().sum() + (cnt - cnt.mean()).square().sum())
        )
        kappa = float(cfg_reg.get("coherence_ratio", 0.75))
        pen_coherence = (
            float(cfg_reg.get("w_coherence", 0.0))
            * m_bar
            * ((cnt - cnt.mean()) - kappa * (cp - cp.mean())).square().sum()
        )
        pen_pr_within = (
            float(cfg_reg.get("w_pr_within", 0.0)) * p_bar * (pr - pr_cm).square().sum()
        )
        pen_pr_between = (
            float(cfg_reg.get("w_pr_between", 0.0))
            * p_bar
            * (pr_cm - pr.mean()).square().sum()
        )

        l2_scale = float(cfg_reg.get("l2_price", 1.0))
        pen_l2 = l2_scale * (
            (pr.square() * price_m).sum()
            + (cp.square() * idx_m).sum()
            + (cnt.square() * idx_m).sum()
            + float(cfg_reg.get("l2_hinge_mult", 1.0)) * (ch.square() * hinge_m).sum()
            + float(cfg_reg.get("l2_cross_mult", 1.0)) * (cr.square() * cross_m).sum()
        )
        return (
            pen_hinge_var
            + pen_cross_var
            + pen_cat_var
            + pen_coherence
            + pen_pr_within
            + pen_pr_between
            + pen_l2
        )

    def newton_polish_counts(self, train: tuple[dict, torch.Tensor]) -> dict:
        """Exact permutation-equivariant Newton count-block polish (matching polish_mission.py)."""
        ctx, y = train
        n_cat = y.sum(-1).long()
        with torch.no_grad():
            for c in range(self.C):
                for k in range(2, self.P):
                    if not (n_cat[:, c] == k + 1).any():
                        self.count_offsets[c, k - 2] = -30.0
        if self.P < 3:
            return {"polished": False}

        tail = self.count_offsets[:, 1:].detach().clone()
        theta = torch.cat((self.count_need.detach(), self.count_offsets[:, 0].detach()))

        def count_loss(v: torch.Tensor) -> torch.Tensor:
            c_need = v[: self.C]
            c_off0 = v[self.C :]
            c_off = torch.cat([c_off0[:, None], tail], dim=-1)
            params_override = {
                "count_need": c_need,
                "count_offsets": c_off,
            }
            return -torch.func.functional_call(self, params_override, (ctx, y)).mean()

        val0 = float(count_loss(theta).detach().item())
        grad = torch.autograd.functional.jacobian(count_loss, theta)
        hess = torch.autograd.functional.hessian(count_loss, theta, vectorize=True)
        eigs = torch.linalg.eigvalsh(hess)
        if float(eigs.min().item()) <= 1e-12:
            return {"polished": False, "min_eigenvalue": float(eigs.min().item())}

        step = torch.linalg.solve(hess, grad)
        frac = 1.0
        best_val = val0
        best_cand = theta
        while frac >= 1.0 / 64.0:
            cand = theta - frac * step
            if (cand >= -15.0).all() and (cand <= 15.0).all():
                cand_val = float(count_loss(cand).detach().item())
                if cand_val <= val0:
                    best_val = cand_val
                    best_cand = cand
                    break
            frac /= 2.0

        with torch.no_grad():
            self.count_need.copy_(best_cand[: self.C])
            self.count_offsets[:, 0].copy_(best_cand[self.C :])
        return {
            "polished": True,
            "nll_before": val0,
            "nll_after": best_val,
            "step_fraction": frac,
            "condition_number": float((eigs.max() / eigs.min()).item()),
        }

    def fit_with_validation_selection(
        self,
        train: tuple[dict, torch.Tensor],
        validation: tuple[dict, torch.Tensor],
        ridge: float = 10.0,
        base_maxiter: int = 400,
        reg_maxiter: int = 400,
        candidates: list[tuple[str, dict | None]] | None = None,
    ) -> dict:
        """Jointly estimate all price and preference blocks over a validation-selected grid of
        permutation-equivariant hierarchical penalties (no category-index ordering or post-hoc edits).
        """
        from .metrics import fit_lbfgsb, certify_fit

        selection_started = time.perf_counter()

        def fit_and_polish(maxiter):
            started = time.perf_counter()
            fit = fit_lbfgsb(self, train, validation, ridge=ridge, maxiter=maxiter)
            polish = self.newton_polish_counts(train)
            certificate = certify_fit(self, train, validation, ridge=ridge)
            # Polishing can disturb other blocks. Refit jointly, once, if needed;
            # never attach a pre-polish certificate to changed saved weights.
            if not certificate["stationarity_passed"]:
                continuation = fit_lbfgsb(
                    self, train, validation, ridge=ridge, maxiter=maxiter
                )
                fit["iterations"] += continuation["iterations"]
                fit["iteration_limit"] += continuation["iteration_limit"]
                fit["optimizer_message"] = continuation["optimizer_message"]
                fit["returned_state"] = "post_polish_joint_continuation"
                fit["post_polish_continuation"] = continuation
                certificate = certify_fit(self, train, validation, ridge=ridge)
            fit.update(certificate)
            fit["seconds"] = time.perf_counter() - started
            fit["polish_record"] = polish
            return fit, polish

        if candidates is None:
            candidates = [
                ("unregularized", None),
                (
                    "mild_hierarchical",
                    {
                        "w_hinge_var": 500.0,
                        "w_cross_var": 800.0,
                        "w_cat_var": 25.0,
                        "w_coherence": 80.0,
                        "w_pr_within": 5.0,
                        "w_pr_between": 5.0,
                        "l2_price": 0.5,
                        "l2_hinge_mult": 3.0,
                        "l2_cross_mult": 120.0,
                    },
                ),
                (
                    "moderate_hierarchical",
                    {
                        "w_hinge_var": 800.0,
                        "w_cross_var": 1200.0,
                        "w_cat_var": 55.0,
                        "w_coherence": 160.0,
                        "w_pr_within": 12.0,
                        "w_pr_between": 8.0,
                        "l2_price": 0.45,
                        "l2_hinge_mult": 3.0,
                        "l2_cross_mult": 200.0,
                    },
                ),
                (
                    "strong_hierarchical",
                    {
                        "w_hinge_var": 1000.0,
                        "w_cross_var": 1500.0,
                        "w_cat_var": 90.0,
                        "w_coherence": 200.0,
                        "w_pr_within": 25.0,
                        "w_pr_between": 10.0,
                        "l2_price": 0.4,
                        "l2_hinge_mult": 3.0,
                        "l2_cross_mult": 250.0,
                    },
                ),
            ]

        self.set_regularization_config(None)
        self.initialize(train)
        unreg_fit, unreg_polish = fit_and_polish(base_maxiter)
        with torch.no_grad():
            unreg_val_ll = float(
                self.log_probability(validation[0], validation[1]).mean().item()
            )
        unreg_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        best_name = "unregularized"
        best_cfg = None
        best_val_ll = unreg_val_ll if unreg_fit["stationarity_passed"] else -np.inf
        best_state = {k: v.clone() for k, v in unreg_state.items()}
        best_fit = dict(unreg_fit)
        best_polish = dict(unreg_polish)

        candidate_records = [
            {
                "name": "unregularized",
                "config": None,
                "validation_ll": unreg_val_ll,
                "raw_projected_gradient": unreg_fit["final_raw_projected_gradient"],
                "scaled_projected_gradient": unreg_fit[
                    "final_scaled_projected_gradient"
                ],
                "stationarity_passed": unreg_fit["stationarity_passed"],
                "iterations": unreg_fit["iterations"],
                "seconds": unreg_fit["seconds"],
                "category_price": self.category_price.detach().cpu().numpy().tolist(),
                "category_hinge": self.category_hinge.detach().cpu().numpy().tolist(),
                "count_price": self.count_price.detach().cpu().numpy().tolist(),
                "category_cross": self.category_cross.detach().cpu().numpy().tolist(),
            }
        ]

        for cand_name, cand_cfg in candidates:
            if cand_cfg is None:
                continue
            self.load_state_dict(unreg_state)
            self.set_regularization_config(cand_cfg)
            c_fit, c_pol = fit_and_polish(reg_maxiter)
            with torch.no_grad():
                c_val_ll = float(
                    self.log_probability(validation[0], validation[1]).mean().item()
                )
            candidate_records.append(
                {
                    "name": cand_name,
                    "config": cand_cfg,
                    "validation_ll": c_val_ll,
                    "raw_projected_gradient": c_fit["final_raw_projected_gradient"],
                    "scaled_projected_gradient": c_fit[
                        "final_scaled_projected_gradient"
                    ],
                    "stationarity_passed": c_fit["stationarity_passed"],
                    "iterations": c_fit["iterations"],
                    "seconds": c_fit["seconds"],
                    "category_price": self.category_price.detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "category_hinge": self.category_hinge.detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "count_price": self.count_price.detach().cpu().numpy().tolist(),
                    "category_cross": self.category_cross.detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            )
            if c_fit["stationarity_passed"] and c_val_ll > best_val_ll:
                best_name = cand_name
                best_cfg = cand_cfg
                best_val_ll = c_val_ll
                best_state = {
                    k: v.detach().clone() for k, v in self.state_dict().items()
                }
                best_fit = dict(c_fit)
                best_polish = dict(c_pol)

        if not np.isfinite(best_val_ll):
            raise RuntimeError(
                "No regularization candidate passed final-weight convergence checks"
            )
        self.load_state_dict(best_state)
        self.set_regularization_config(best_cfg)
        selected_seconds = best_fit["seconds"]
        best_fit.update(certify_fit(self, train, validation, ridge=ridge))
        best_fit["selected_candidate_seconds"] = selected_seconds
        best_fit["seconds"] = time.perf_counter() - selection_started
        best_fit["selection_includes_initialization_all_candidates_and_polishing"] = (
            True
        )
        best_fit["selected_regularization_config"] = best_cfg
        return {
            "total_selection_seconds": best_fit["seconds"],
            "selected_candidate": best_name,
            "selected_config": best_cfg,
            "selected_validation_ll": best_val_ll,
            "unregularized_validation_ll": unreg_val_ll,
            "unregularized_state_dict": unreg_state,
            "unregularized_fit_record": unreg_fit,
            "selected_fit_record": best_fit,
            "selected_polish_record": best_polish,
            "candidate_records": candidate_records,
            "permutation_equivariant": True,
            "category_ordering_used": False,
            "post_hoc_shrinkage_used": False,
        }

    @torch.no_grad()
    def exact_mass(self, ctx: dict, sets: torch.Tensor) -> torch.Tensor:
        eta, count, subset, normalizer, mixture, _ = self.parts(
            ctx, include_subset_table=True
        )
        sets = sets.reshape(-1, self.C, self.P)
        n = sets.sum(-1).long()
        code = (sets.long() * (2 ** torch.arange(self.P, device=sets.device))).sum(-1)
        output = -normalizer[:, :, None].expand(-1, -1, len(sets)).clone()
        for c in range(self.C):
            output += torch.where(
                n[:, c][None, None] > 0,
                F.logsigmoid(eta[:, :, c, None])
                + count[:, :, c, (n[:, c] - 1).clamp(min=0)]
                + subset[:, :, c, code[:, c]],
                F.logsigmoid(-eta[:, :, c, None]),
            )
        return torch.logsumexp(output + mixture[None, :, None], 1).exp()

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        eta, count, _, _, mixture, u = self.parts(ctx, include_subset_table=False)
        generator = torch.Generator().manual_seed(seed)
        mission = torch.multinomial(
            mixture.exp().expand(len(eta), -1), 1, generator=generator
        ).ravel()
        row = torch.arange(len(eta))
        eta, count, u = eta[row, mission], count[row, mission], u[row, mission]
        nonempty_suffix = torch.log(
            (-torch.expm1(F.logsigmoid(-eta).flip(-1).cumsum(-1).flip(-1))).clamp(
                min=1e-30
            )
        )
        present = torch.zeros_like(eta, dtype=torch.bool)
        active = torch.zeros(len(eta), dtype=torch.bool)
        for c in range(self.C):
            probability = torch.where(
                active,
                eta[:, c].sigmoid(),
                (F.logsigmoid(eta[:, c]) - nonempty_suffix[:, c]).exp(),
            )
            present[:, c] = torch.rand(
                len(eta), generator=generator
            ) < probability.clamp(0, 1)
            active |= present[:, c]
        n = (
            torch.multinomial(
                count.exp().reshape(-1, self.P), 1, generator=generator
            ).reshape(-1, self.C)
            + 1
        )
        uniform = torch.rand(u.shape, generator=generator, dtype=u.dtype).clamp(
            1e-15, 1 - 1e-15
        )
        order = (u - (-uniform.log()).log()).argsort(-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            -1, order, torch.arange(self.P, device=u.device).expand_as(order)
        )
        return ((ranks < n[..., None]) & present[..., None]).double()
