"""Proposed Whole-Law Basket Models:
1. CompositeWholeLawEnergy: Exact-normalized whole-basket mixture combining
   (a) category-frontier DP energy with nonparametric global count potential g(N),
       shopper count slope s_h*(N-1), and exact size-1 quick-trip utility stratum W_1,
   (b) exact-partition pre-trip category mission mixture, and
   (c) exact-partition product bundle mission mixture.
2. CalibratedMissionLaw: Differentiable exponential-race PL mission law with
   non-negative hierarchical cross-price pooling.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .race_pl import race_subset_log_probabilities


def count_pairs(n: torch.Tensor) -> torch.Tensor:
    n = n.double()
    return n * (n - 1) / 2.0


def log_esp(u: torch.Tensor, max_deg: int | None = None) -> torch.Tensor:
    """All elementary symmetric polynomial log-coefficients up to max_deg."""
    if u.ndim == 3:
        B, C, P = u.shape
        flat = log_esp(u.reshape(B * C, P), max_deg=max_deg)
        return flat.reshape(B, C, -1)
    coefficients = u.new_zeros((len(u), 1))
    for j in range(u.shape[1]):
        absent = torch.cat((coefficients, u.new_full((len(u), 1), -torch.inf)), -1)
        present = torch.cat(
            (u.new_full((len(u), 1), -torch.inf), coefficients + u[:, j : j + 1]), -1
        )
        coefficients = torch.logaddexp(absent, present)
        if max_deg is not None and coefficients.shape[1] > max_deg + 1:
            coefficients = coefficients[:, : max_deg + 1]
    return coefficients


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
        value = (y * utility).sum(-1) + (count_pairs(counts) * self.category_count).sum(-1)
        value += count_pairs(counts.sum(-1)) * self.global_count[0]
        for k, (i, j) in enumerate(self.edges):
            value = value + self.link[k] * y[:, i] * y[:, j]
        return value

    def partition(self, utility: torch.Tensor, keep_paths: bool = False):
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
            maximum = values.new_full((len(utility), len(topology["states"])), -torch.inf)
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
            candidate_weights = candidate_weights.masked_fill(candidates < 0, -torch.inf)
            selected = torch.multinomial(
                candidate_weights.softmax(-1), 1, generator=generator
            ).ravel()
            transition = candidates[row, selected]
            mask = topology["mask"][transition]
            output[:, topology["interface"]] = topology["bits"][mask]
            remaining = topology["extra"][transition].clone()
            free = topology["free"]
            suffix = [None] * (len(free) + 1)
            suffix[-1] = utility.new_zeros((len(utility), 1))
            for k in range(len(free) - 1, -1, -1):
                coefficients = suffix[k + 1]
                absent = torch.cat(
                    (coefficients, utility.new_full((len(utility), 1), -torch.inf)), -1
                )
                present = torch.cat(
                    (
                        utility.new_full((len(utility), 1), -torch.inf),
                        coefficients + utility[:, free[k] : free[k] + 1],
                    ),
                    -1,
                )
                suffix[k] = torch.logaddexp(absent, present)
            for k, j in enumerate(free):
                n = remaining
                denominator = suffix[k][row, n]
                index = (n - 1).clamp(min=0, max=suffix[k + 1].shape[1] - 1)
                numerator = utility[:, j] + suffix[k + 1][row, index]
                probability = (numerator - denominator).exp().clamp(0, 1)
                probability = torch.where(n == 0, 0.0, probability)
                probability = torch.where(n == len(free) - k, 1.0, probability)
                take = torch.rand(len(utility), generator=generator) < probability
                output[:, j] = take
                remaining -= take.long()
            assert (remaining == 0).all()
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
    ):
        super().__init__()
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.users = int(config["users"])
        self.config = {
            **config,
            "model_family": "composite_whole_law_nonparametric_count_and_missions",
            "category_missions": int(len(M_cat_init)),
            "product_missions": int(len(M_prod_fixed)),
        }
        self.re_model = re_model
        for p in self.re_model.parameters():
            p.requires_grad_(False)

        self.K_c = len(M_cat_init)
        self.K_p = len(M_prod_fixed)
        self.K_tot = 1 + self.K_c + self.K_p
        self.register_buffer("M_prod_fixed", M_prod_fixed.clone().double())

        subsets = torch.tensor(
            [[int((m >> j) & 1) for j in range(self.P)] for m in range(1 << self.P)],
            dtype=torch.float64,
        )
        sub_ne = subsets[1:]
        self.register_buffer("sub_ne", sub_ne)
        cat_edge_mat = torch.zeros((self.C, (1 << self.P) - 1), dtype=torch.float64)
        for k, (i, j) in enumerate(re_model.edges):
            if i // self.P == j // self.P:
                c = i // self.P
                li, lj = i % self.P, j % self.P
                cat_edge_mat[c] += float(re_model.link[k].detach()) * (
                    sub_ne[:, li] * sub_ne[:, lj]
                )
        n_pairs = sub_ne.sum(-1) * (sub_ne.sum(-1) - 1.0) / 2.0
        base_cat_h = (
            re_model.category_count.detach()[:, None] * n_pairs[None, :] + cat_edge_mat
        )
        self.register_buffer("base_cat_h", base_cat_h)

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
        self.cat_shift = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.house_cat = nn.Parameter(
            torch.zeros((self.users, self.C), dtype=torch.float64)
        )
        self.price_cat = nn.Parameter(torch.zeros((self.J, self.C), dtype=torch.float64))

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

    def within_category_energy(self, y_flat: torch.Tensor, u_base: torch.Tensor) -> torch.Tensor:
        counts = y_flat.reshape(-1, self.C, self.P).sum(-1)
        val = (y_flat * u_base).sum(-1) + (
            count_pairs(counts) * self.re_model.category_count
        ).sum(-1)
        for k, (i, j) in enumerate(self.re_model.edges):
            if i // self.P == j // self.P:
                val = val + self.re_model.link[k] * y_flat[:, i] * y_flat[:, j]
        return val

    def precompute_split(self, ctx, y=None):
        """Precompute immutable base parent terms for fast exact optimization."""
        with torch.no_grad():
            u_base = self.re_model.utilities(ctx)
            parts = []
            for i in range(0, len(u_base), 4096):
                parts.append(
                    self.re_model.partition(u_base[i : i + 4096], keep_paths=True)[3]
                )
            final_base = torch.cat(parts, 0)
            fc = self.re_model.final_counts
            quad = self.re_model.global_count[0].detach() * count_pairs(fc)
            final_unquad = final_base - quad[None, :]
            cols_by_N = [
                torch.full((len(u_base),), -torch.inf, dtype=torch.float64, device=u_base.device)
            ]
            for n in range(1, self.J + 1):
                idx = (fc == n).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    cols_by_N.append(
                        torch.full((len(u_base),), -torch.inf, dtype=torch.float64, device=u_base.device)
                    )
                elif len(idx) == 1:
                    cols_by_N.append(final_unquad[:, idx[0]])
                else:
                    cols_by_N.append(torch.logsumexp(final_unquad[:, idx], dim=-1))
            W_base = torch.stack(cols_by_N, dim=-1)

            u_cp = u_base.reshape(-1, self.C, self.P)
            log_W_cat_base = torch.logsumexp(
                u_cp @ self.sub_ne.T + self.base_cat_h[None, :, :], dim=-1
            )
            m_prod_sub = (
                self.M_prod_fixed.reshape(self.K_p, self.C, self.P) @ self.sub_ne.T
                + self.base_cat_h[None, :, :]
            )
            prod_parts = []
            for i in range(0, len(u_base), 4096):
                ub = u_cp[i : i + 4096] @ self.sub_ne.T
                prod_parts.append(
                    torch.logsumexp(
                        ub[:, None, :, :] + m_prod_sub[None, :, :, :], dim=-1
                    )
                )
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
            return cache

    def forward_cached(self, ctx, cache):
        B = len(ctx["house"])
        H = ctx["house"]
        d = ctx["dlp"].reshape(B, self.J)
        u_base = cache["u_base"]
        W_base = cache["W_base"]
        log_W_cat_base = cache["log_W_cat_base"]
        log_W_prod_base = cache["log_W_prod_base"]
        y_flat = cache["y_flat"]
        N_obs = cache["N_obs"]
        e_base = cache["e_base"]
        e_within = cache["e_within"]
        present = cache["present"]
        y_m_prod = cache["y_m_prod"]

        counts = torch.arange(self.J + 1, dtype=torch.float64)
        n_minus_1 = (counts - 1.0).clamp(min=0.0)
        g = self.build_g()
        dx_1 = (
            self.delta_1
            + self.house_1[H]
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
        pi_logits = self.mix_logits[None, :] + self.house_mix[H] + d @ self.price_mix
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return torch.logsumexp(log_pi + all_comp_ll, dim=-1)

    def log_probability(self, ctx, y):
        cache = self.precompute_split(ctx, y)
        return self.forward_cached(ctx, cache)

    @torch.no_grad()
    def sample(self, ctx, seed=0):
        """Exact rejection-free ancestral mixture + backward DP sampler."""
        B = len(ctx["house"])
        H = ctx["house"]
        d = ctx["dlp"].reshape(B, self.J)
        u_base = self.re_model.utilities(ctx)
        _, _, paths, final_base = self.re_model.partition(u_base, keep_paths=True)
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
                    torch.full((B,), -torch.inf, dtype=torch.float64, device=u_base.device)
                )
            elif len(idx) == 1:
                cols_by_N.append(final_unquad[:, idx[0]])
            else:
                cols_by_N.append(torch.logsumexp(final_unquad[:, idx], dim=-1))
        W_base = torch.stack(cols_by_N, dim=-1)
        counts = torch.arange(self.J + 1, dtype=torch.float64)
        n_minus_1 = (counts - 1.0).clamp(min=0.0)

        pi_logits = self.mix_logits[None, :] + self.house_mix[H] + d @ self.price_mix
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
            logits_0 = W_mod + g[None, :] + self.s_house[H0][:, None] * n_minus_1[None, :]
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
                # Draw final DP state conditioned on final_counts == req_n
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
                    suffix = [None] * (len(free) + 1)
                    suffix[-1] = u_gt.new_zeros((len(u_gt), 1))
                    for k in range(len(free) - 1, -1, -1):
                        coeff = suffix[k + 1]
                        absent = torch.cat(
                            (coeff, u_gt.new_full((len(u_gt), 1), -torch.inf)), -1
                        )
                        present_t = torch.cat(
                            (
                                u_gt.new_full((len(u_gt), 1), -torch.inf),
                                coeff + u_gt[:, free[k] : free[k] + 1],
                            ),
                            -1,
                        )
                        suffix[k] = torch.logaddexp(absent, present_t)
                    for k, j in enumerate(free):
                        n_req = rem
                        denom = suffix[k][row_arange, n_req]
                        idx_next = (n_req - 1).clamp(
                            min=0, max=suffix[k + 1].shape[1] - 1
                        )
                        numer = u_gt[:, j] + suffix[k + 1][row_arange, idx_next]
                        prob = (numer - denom).exp().clamp(0, 1)
                        prob = torch.where(n_req == 0, 0.0, prob)
                        prob = torch.where(n_req == len(free) - k, 1.0, prob)
                        take = (
                            torch.rand(len(u_gt), generator=gen, dtype=u_gt.dtype)
                            < prob
                        )
                        y_gt[:, j] = take.double()
                        rem -= take.long()
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
            sub_logits = u_eff @ self.sub_ne.T + self.base_cat_h[None, :, :]
            log_W_m = torch.logsumexp(sub_logits, dim=-1) + d_cat + cat_shift_m

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

            sub_probs = sub_logits.softmax(dim=-1)
            drawn_sub = torch.multinomial(
                sub_probs.reshape(-1, (1 << self.P) - 1), 1, generator=gen
            ).reshape(len(idx_m), self.C)
            chosen_bits = self.sub_ne[drawn_sub]
            output[idx_m] = chosen_bits * pres_draw[:, :, None].double()

        assert output.any(dim=(1, 2)).all(), "Nonempty conditioning failed"
        return output


class CalibratedMissionLaw(nn.Module):
    """Joint pre-trip mission + count + differentiable exponential-race PL law."""

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
        self.count_offsets = parameter((self.C, 2))
        self.mixture_logit = parameter((1,))

    def parts(self, ctx: dict):
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
        count_logits = torch.stack(
            (
                torch.zeros_like(count_eta),
                count_eta,
                2 * count_eta + self.count_offsets[:, 0],
                3 * count_eta + self.count_offsets[:, 1],
            ),
            -1,
        )
        count = F.log_softmax(count_logits, -1)
        u = (self.lam + self.house[ctx["house"]] - self.price * d)[:, None]
        u = u + signs[..., None] * self.product_contrast
        subset = race_subset_log_probabilities(u, n_nodes=64)
        empty = F.logsigmoid(-eta).sum(-1)
        log_nonempty = torch.log(-torch.expm1(empty))
        log_mixture = torch.stack(
            (F.logsigmoid(self.mixture_logit[0]), F.logsigmoid(-self.mixture_logit[0]))
        )
        return eta, count, subset, log_nonempty, log_mixture, u

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        eta, count, subset, normalizer, mixture, _ = self.parts(ctx)
        n = y.sum(-1).long()
        present = n > 0
        code = (y.long() * (2 ** torch.arange(self.P, device=y.device))).sum(-1)
        choice = subset.gather(-1, code[:, None, :, None].expand(-1, 2, -1, 1)).squeeze(-1)
        size = count.gather(
            -1, (n - 1).clamp(min=0)[:, None, :, None].expand(-1, 2, -1, 1)
        ).squeeze(-1)
        participation = torch.where(
            present[:, None], F.logsigmoid(eta), F.logsigmoid(-eta)
        )
        component = (
            participation + torch.where(present[:, None], size + choice, 0.0)
        ).sum(-1) - normalizer
        result = torch.logsumexp(component + mixture, -1)
        return result.masked_fill(n.sum(-1) == 0, -torch.inf)

    @torch.no_grad()
    def exact_mass(self, ctx: dict, sets: torch.Tensor) -> torch.Tensor:
        eta, count, subset, normalizer, mixture, _ = self.parts(ctx)
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
        eta, count, _, _, mixture, u = self.parts(ctx)
        generator = torch.Generator().manual_seed(seed)
        mission = torch.multinomial(
            mixture.exp().expand(len(eta), -1), 1, generator=generator
        ).ravel()
        row = torch.arange(len(eta))
        eta, count, u = eta[row, mission], count[row, mission], u[row, mission]
        nonempty_suffix = torch.log(
            -torch.expm1(F.logsigmoid(-eta).flip(-1).cumsum(-1).flip(-1))
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
        ranks.scatter_(-1, order, torch.arange(self.P).expand_as(order))
        return ((ranks < n[..., None]) & present[..., None]).double()
