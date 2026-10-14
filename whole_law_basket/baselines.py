"""Comprehensive Baseline Models for Whole-Law Basket Choice and Generation.

Includes 10 distinct baseline families spanning econometric, hierarchical ESP,
latent-mission, graphical/autoregressive, low-rank bilinear autoregressive,
finite Gaussian factor mixture, and neural NADE architectures:
  1. PooledPopularity (popularity)
  2. NoPriceIndependent (no_price)
  3. PersonalizedIndependent (independent_parent)
  4. SizeAwareESP (size_aware_esp)
  5. IndependentCategoriesESP (independent_categories_esp)
  6. LatentMissionESP (mission_esp)
  7. PairwiseBinaryAR (pairwise_ar)
  8. LowRankBilinearAR (low_rank_bilinear_ar; fixed-order bilinear AR, alias ShopperFactorAR)
  9. GaussianFactorMixture (gaussian_factor_mixture; finite Gaussian-node mixture, alias LogisticNormalFactorMixture)
  10. FastNeuralNADE (neural_nade)
  (Plus StructuredEnergy in proposed_model.py as the unstratified energy baseline.)

Every model implements exact nonempty-basket normalization over all 2^J - 1
nonempty baskets and an exact, rejection-free `.sample(ctx, seed)` method.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .proposed_model import log_esp


def _nonempty_ar_normalizer(empty_logits: torch.Tensor) -> torch.Tensor:
    """Exact log P(y != 0) for an autoregressive binary chain from its empty-prefix logits."""
    no = F.logsigmoid(-empty_logits)
    prefix = no.cumsum(-1) - no
    return torch.logsumexp(F.logsigmoid(empty_logits) + prefix, dim=-1)


def _sample_nonempty_bernoulli(
    u: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Exact rejection-free sequential sampling from independent Bernoulli(sigmoid(u)) | y != 0."""
    B, J = u.shape
    suffix = F.softplus(u.flip(-1)).cumsum(-1).flip(-1)
    normalizer = torch.log((-torch.expm1(-suffix)).clamp(min=1e-30))
    output = torch.zeros_like(u)
    active = torch.zeros(B, dtype=torch.bool, device=u.device)
    for j in range(J):
        p = torch.where(
            active,
            u[:, j].sigmoid(),
            (F.logsigmoid(u[:, j]) - normalizer[:, j]).exp(),
        ).clamp(0, 1)
        take = torch.rand(B, generator=generator, device=u.device, dtype=u.dtype) < p
        output[:, j] = take.to(u.dtype)
        active |= take
    return output


def _sample_esp_subset(
    u: torch.Tensor, sizes: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Exact backward-DP subset draw of size `sizes[b]` from items with log-weights `u[b]`."""
    B, J = u.shape
    suffix = [None] * (J + 1)
    suffix[-1] = u.new_zeros((B, 1))
    max_n = int(sizes.max().item())
    for k in range(J - 1, -1, -1):
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
    for k in range(J):
        n = remaining
        denom = suffix[k][row, n.clamp(max=suffix[k].shape[1] - 1)]
        idx = (n - 1).clamp(min=0, max=suffix[k + 1].shape[1] - 1)
        num = u[:, k] + suffix[k + 1][row, idx]
        prob = (num - denom).exp().clamp(0, 1)
        prob = torch.where(n == 0, 0.0, prob)
        prob = torch.where(n == J - k, 1.0, prob)
        take = torch.rand(B, generator=generator, device=u.device, dtype=u.dtype) < prob
        output[:, k] = take.to(u.dtype)
        remaining -= take.long()
    return output


class PooledPopularity(nn.Module):
    """Baseline 1: Pooled conditional independent products (item intercepts only)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "popularity"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))

    def utilities(self, ctx: dict) -> torch.Tensor:
        B = len(ctx["house"])
        return self.lam[None, :].expand(B, -1)

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        a = F.softplus(self.lam).sum()
        norm = a + torch.log((-torch.expm1(-a)).clamp(min=1e-30))
        return y.reshape(-1, self.J).double() @ self.lam - norm

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u = self.utilities(ctx)
        gen = torch.Generator().manual_seed(seed)
        return _sample_nonempty_bernoulli(u, gen).reshape(-1, self.C, self.P)


class NoPriceIndependent(nn.Module):
    """Baseline 2: Personalized independent logit with price sensitivity frozen at zero."""

    def __init__(self, config: dict, has_baseline: bool = False):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "no_price"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((int(config["users"]), self.J), dtype=torch.float64),
            requires_grad=not has_baseline,
        )

    def utilities(self, ctx: dict) -> torch.Tensor:
        B = len(ctx["house"])
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        u = base + self.lam
        if self.house.requires_grad or bool(self.house.abs().max() > 0):
            u = u + self.house[ctx["house"]]
        return u

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        u = self.utilities(ctx)
        a = F.softplus(u).sum(-1)
        return (
            (u * y.reshape(-1, self.J).double()).sum(-1)
            - a
            - torch.log((-torch.expm1(-a)).clamp(min=1e-30))
        )

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u = self.utilities(ctx)
        gen = torch.Generator().manual_seed(seed)
        return _sample_nonempty_bernoulli(u, gen).reshape(-1, self.C, self.P)


class PersonalizedIndependent(nn.Module):
    """Baseline 3: Full personalized independent binary logit conditioned on nonempty basket."""

    def __init__(self, config: dict, price: torch.Tensor | None = None):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "independent_parent"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((int(config["users"]), self.J), dtype=torch.float64)
        )
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.price = nn.Parameter(init_p)

    def utilities(self, ctx: dict) -> torch.Tensor:
        B = len(ctx["house"])
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        d = ctx["dlp"].reshape(B, self.J)
        return base + self.lam + self.house[ctx["house"]] - self.price * d

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        u = self.utilities(ctx)
        a = F.softplus(u).sum(-1)
        return (
            (u * y.reshape(-1, self.J).double()).sum(-1)
            - a
            - torch.log((-torch.expm1(-a)).clamp(min=1e-30))
        )

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u = self.utilities(ctx)
        gen = torch.Generator().manual_seed(seed)
        return _sample_nonempty_bernoulli(u, gen).reshape(-1, self.C, self.P)


class SizeAwareESP(nn.Module):
    """Baseline 4: Predicted basket size P(N | h, p) + conditional ESP subset choice."""

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        train_y: torch.Tensor | None = None,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "size_aware_esp"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        if train_y is not None:
            counts = (
                np.bincount(
                    train_y.reshape(-1, self.J).sum(-1).numpy().astype(int),
                    minlength=self.J + 1,
                )[1:]
                + 0.5
            )
            init_lam = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float64)
        else:
            init_lam = torch.zeros(self.J, dtype=torch.float64)
        self.size_lam = nn.Parameter(init_lam - init_lam.mean())
        self.need = nn.Parameter(torch.zeros(int(config["users"]), dtype=torch.float64))
        self.price_slope = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self._within_cache = {}

    @torch.no_grad()
    def fit_information(self, ctx: dict) -> dict:
        """Exact diagonal Fisher information for the conditional size softmax.

        A shopper's slope multiplies N-1, so its curvature is frequency times
        Var(N), rather than the Bernoulli curvature used for item intercepts.
        This changes optimizer units only, preserving the likelihood and prior.
        """
        _, log_q = self.terms(ctx)
        q = log_q.exp()
        grid = torch.arange(self.J, dtype=q.dtype, device=q.device)
        variance = (q * grid.square()).sum(-1) - (q * grid).sum(-1).square()
        n = len(q)
        return {
            "size_lam": (q * (1.0 - q)).mean(0),
            "need": torch.bincount(
                ctx["house"], weights=variance, minlength=len(self.need)
            )
            / n,
            "price_slope": (variance * ctx["dlp"].reshape(n, -1).mean(-1).square())
            .mean()
            .reshape(1),
        }

    def terms(self, ctx: dict):
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.J)
        if self.parent is not None:
            with torch.no_grad():
                u = self.parent.utilities(ctx).reshape(B, self.J)
        else:
            base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
            u = base - self.base_price * d
        n_grid = torch.arange(self.J, dtype=torch.float64, device=d.device)
        slope = self.need[ctx["house"]] + self.price_slope[0] * d.mean(-1)
        q = F.log_softmax(
            self.size_lam[None, :] + slope[:, None] * n_grid[None, :], dim=-1
        )
        return u, q

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if not grad_in and cache_key in self._within_cache:
            within, d_mean, n_idx = self._within_cache[cache_key]
            n_grid = torch.arange(self.J, dtype=torch.float64, device=d_mean.device)
            slope = self.need[ctx["house"]] + self.price_slope[0] * d_mean
            q = F.log_softmax(
                self.size_lam[None, :] + slope[:, None] * n_grid[None, :], dim=-1
            )
            return within + q.gather(1, n_idx).ravel()

        u, q = self.terms(ctx)
        y_flat = y.reshape(-1, self.J).double()
        n = y_flat.sum(-1).long()
        if not grad_in:
            with torch.no_grad():
                e = log_esp(u, max_deg=int(n.max().item()))
                within = (u * y_flat).sum(-1) - e.gather(1, n[:, None]).ravel()
                d_mean = ctx["dlp"].reshape(len(y_flat), self.J).mean(-1)
                n_idx = (n - 1)[:, None]
                self._within_cache[cache_key] = (within, d_mean, n_idx)
            return within + q.gather(1, n_idx).ravel()
        e = log_esp(u, max_deg=int(n.max().item()))
        within = (u * y_flat).sum(-1) - e.gather(1, n[:, None]).ravel()
        return within + q.gather(1, (n - 1)[:, None]).ravel()

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u, q = self.terms(ctx)
        gen = torch.Generator().manual_seed(seed)
        sizes = torch.multinomial(q.exp(), 1, generator=gen).ravel() + 1
        out = _sample_esp_subset(u, sizes, gen)
        return out.reshape(-1, self.C, self.P)


class IndependentCategoriesESP(nn.Module):
    """Baseline 5: Category-specific participation + within-category ESP count curvature."""

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "independent_categories_esp"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        self.item_lam = nn.Parameter(
            torch.zeros((self.C, self.P), dtype=torch.float64), requires_grad=False
        )
        self.kappa = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.rho = nn.Parameter(torch.full((self.C,), 0.8, dtype=torch.float64))
        self.nu = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.tau = nn.Parameter(torch.zeros(self.C, dtype=torch.float64))
        self.house_cat = nn.Parameter(
            torch.zeros((int(config["users"]), self.C), dtype=torch.float64)
        )
        self._split_cache = {}

    def _base_u_and_esp(self, ctx: dict):
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.C, self.P)
        if self.parent is not None:
            with torch.no_grad():
                u0 = self.parent.utilities(ctx).reshape(B, self.C, self.P)
        else:
            base = (
                ctx["baseline"].reshape(B, self.C, self.P) if "baseline" in ctx else 0.0
            )
            u0 = base - self.base_price.reshape(1, self.C, self.P) * d
        u = u0 + self.item_lam[None, :, :]
        return u, log_esp(u)

    def terms(self, ctx: dict):
        u, esp = self._base_u_and_esp(ctx)
        n = torch.arange(1, self.P + 1, dtype=torch.float64, device=u.device)
        penalty = self.rho[:, None] * (n * (n - 1) / 2.0) - self.nu[:, None] * (n - 1.0)
        a = esp[..., 1:] - penalty[None, :, :]
        iv = torch.logsumexp(a, dim=-1)
        eta = (
            self.kappa[None, :]
            + self.house_cat[ctx["house"]]
            + F.softplus(self.tau)[None, :] * iv
        )
        return u, esp, F.log_softmax(a, dim=-1), eta

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if not grad_in and not self.item_lam.requires_grad:
            if cache_key not in self._split_cache:
                with torch.no_grad():
                    u, esp = self._base_u_and_esp(ctx)
                    y_cat = y.reshape(-1, self.C, self.P).double()
                    n = y_cat.sum(-1).long()
                    x = (n > 0).double()
                    u_minus_esp = (u * y_cat).sum(-1) - esp.gather(
                        -1, n[..., None]
                    ).squeeze(-1)
                    n_idx = (n - 1).clamp(min=0)[..., None]
                    self._split_cache[cache_key] = (esp[..., 1:], x, u_minus_esp, n_idx)
            esp_pos, x, u_minus_esp, n_idx = self._split_cache[cache_key]
            n_grid = torch.arange(
                1, self.P + 1, dtype=torch.float64, device=esp_pos.device
            )
            penalty = self.rho[:, None] * (n_grid * (n_grid - 1) / 2.0) - self.nu[
                :, None
            ] * (n_grid - 1.0)
            a = esp_pos - penalty[None, :, :]
            count_lp = F.log_softmax(a, dim=-1)
            iv = torch.logsumexp(a, dim=-1)
            eta = (
                self.kappa[None, :]
                + self.house_cat[ctx["house"]]
                + F.softplus(self.tau)[None, :] * iv
            )
            cat_ll = (x * F.logsigmoid(eta) + (1.0 - x) * F.logsigmoid(-eta)).sum(
                -1
            ) - torch.log((-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30))
            within = u_minus_esp + count_lp.gather(-1, n_idx).squeeze(-1)
            return cat_ll + (within * x).sum(-1)

        u, esp, count_lp, eta = self.terms(ctx)
        y_cat = y.reshape(-1, self.C, self.P).double()
        n = y_cat.sum(-1).long()
        x = (n > 0).double()
        cat_ll = (x * F.logsigmoid(eta) + (1.0 - x) * F.logsigmoid(-eta)).sum(
            -1
        ) - torch.log((-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30))
        within = (
            (u * y_cat).sum(-1)
            - esp.gather(-1, n[..., None]).squeeze(-1)
            + count_lp.gather(-1, (n - 1).clamp(min=0)[..., None]).squeeze(-1)
        )
        return cat_ll + (within * x).sum(-1)

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u, _, count_lp, eta = self.terms(ctx)
        gen = torch.Generator().manual_seed(seed)
        B = len(eta)
        pres = _sample_nonempty_bernoulli(eta, gen)
        out = torch.zeros((B, self.C, self.P), dtype=torch.float64, device=u.device)
        for c in range(self.C):
            active_rows = (pres[:, c] > 0.5).nonzero(as_tuple=True)[0]
            if len(active_rows) > 0:
                sizes_c = (
                    torch.multinomial(
                        count_lp[active_rows, c].exp(), 1, generator=gen
                    ).ravel()
                    + 1
                )
                out[active_rows, c] = _sample_esp_subset(
                    u[active_rows, c], sizes_c, gen
                )
        return out


class LatentMissionESP(nn.Module):
    """Baseline 6: K=2 latent-mission mixture of category-independent ESP laws (MissionESP)."""

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        K: int = 2,
        seed: int = 0,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {**config, "benchmark_baseline": f"mission_esp_K{K}"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.K = K
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        g = torch.Generator().manual_seed(seed)
        self.taste = nn.Parameter(
            0.05 * torch.randn((K, self.C, self.P), generator=g, dtype=torch.float64),
            requires_grad=False,
        )
        self.category = nn.Parameter(
            0.15 * torch.randn((K, self.C), generator=g, dtype=torch.float64)
        )
        self.nu = nn.Parameter(torch.zeros((K, self.C), dtype=torch.float64))
        self.rho = nn.Parameter(torch.full((K, self.C), 0.8, dtype=torch.float64))
        self.tau = nn.Parameter(torch.zeros((K, self.C), dtype=torch.float64))
        self.weights = nn.Parameter(
            torch.zeros((int(config["users"]), K), dtype=torch.float64)
        )
        self._split_cache = {}

    def terms(self, ctx: dict, m: int):
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.C, self.P)
        if self.parent is not None:
            with torch.no_grad():
                u0 = self.parent.utilities(ctx).reshape(B, self.C, self.P)
        else:
            base = (
                ctx["baseline"].reshape(B, self.C, self.P) if "baseline" in ctx else 0.0
            )
            u0 = base - self.base_price.reshape(1, self.C, self.P) * d
        u = u0 + self.taste[m][None, :, :]
        esp = log_esp(u)
        n = torch.arange(1, self.P + 1, dtype=torch.float64, device=u.device)
        penalty = self.rho[m, :, None] * (n * (n - 1) / 2.0) - self.nu[m, :, None] * (
            n - 1.0
        )
        a = esp[..., 1:] - penalty[None, :, :]
        iv = torch.logsumexp(a, dim=-1)
        eta = self.category[m][None, :] + F.softplus(self.tau[m])[None, :] * iv
        return u, esp, F.log_softmax(a, dim=-1), eta

    def component_ll(self, ctx: dict, y_cat: torch.Tensor, m: int) -> torch.Tensor:
        u, esp, count_lp, eta = self.terms(ctx, m)
        n = y_cat.sum(-1).long()
        x = (n > 0).double()
        cat_ll = (x * F.logsigmoid(eta) + (1.0 - x) * F.logsigmoid(-eta)).sum(
            -1
        ) - torch.log((-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30))
        within = (
            (u * y_cat).sum(-1)
            - esp.gather(-1, n[..., None]).squeeze(-1)
            + count_lp.gather(-1, (n - 1).clamp(min=0)[..., None]).squeeze(-1)
        )
        return cat_ll + (within * x).sum(-1)

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if not grad_in and not self.taste.requires_grad:
            if cache_key not in self._split_cache:
                with torch.no_grad():
                    B = len(ctx["house"])
                    d = ctx["dlp"].reshape(B, self.C, self.P)
                    if self.parent is not None:
                        u0 = self.parent.utilities(ctx).reshape(B, self.C, self.P)
                    else:
                        base = (
                            ctx["baseline"].reshape(B, self.C, self.P)
                            if "baseline" in ctx
                            else 0.0
                        )
                        u0 = base - self.base_price.reshape(1, self.C, self.P) * d
                    y_cat = y.reshape(B, self.C, self.P).double()
                    n = y_cat.sum(-1).long()
                    x = (n > 0).double()
                    n_idx = (n - 1).clamp(min=0)[..., None]
                    per_m = []
                    for m in range(self.K):
                        um = u0 + self.taste[m][None, :, :]
                        espm = log_esp(um)
                        u_minus_esp = (um * y_cat).sum(-1) - espm.gather(
                            -1, n[..., None]
                        ).squeeze(-1)
                        per_m.append((espm[..., 1:], u_minus_esp))
                    self._split_cache[cache_key] = (x, n_idx, per_m)
            x, n_idx, per_m = self._split_cache[cache_key]
            n_grid = torch.arange(1, self.P + 1, dtype=torch.float64, device=x.device)
            comps = []
            for m in range(self.K):
                esp_pos, u_minus_esp = per_m[m]
                penalty = self.rho[m, :, None] * (
                    n_grid * (n_grid - 1) / 2.0
                ) - self.nu[m, :, None] * (n_grid - 1.0)
                a = esp_pos - penalty[None, :, :]
                count_lp = F.log_softmax(a, dim=-1)
                iv = torch.logsumexp(a, dim=-1)
                eta = self.category[m][None, :] + F.softplus(self.tau[m])[None, :] * iv
                cat_ll = (x * F.logsigmoid(eta) + (1.0 - x) * F.logsigmoid(-eta)).sum(
                    -1
                ) - torch.log(
                    (-torch.expm1(F.logsigmoid(-eta).sum(-1))).clamp(min=1e-30)
                )
                within = u_minus_esp + count_lp.gather(-1, n_idx).squeeze(-1)
                comps.append(cat_ll + (within * x).sum(-1))
            w = F.log_softmax(self.weights[ctx["house"]], dim=-1)
            return torch.logsumexp(torch.stack(comps, dim=-1) + w, dim=-1)

        y_cat = y.reshape(-1, self.C, self.P).double()
        w = F.log_softmax(self.weights[ctx["house"]], dim=-1)
        comp = torch.stack(
            [self.component_ll(ctx, y_cat, m) for m in range(self.K)], dim=-1
        )
        return torch.logsumexp(comp + w, dim=-1)

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        B = len(ctx["house"])
        w = F.softmax(self.weights[ctx["house"]], dim=-1)
        chosen_m = torch.multinomial(w, 1, generator=gen).ravel()
        out = torch.zeros((B, self.C, self.P), dtype=torch.float64)
        for m in range(self.K):
            rows = (chosen_m == m).nonzero(as_tuple=True)[0]
            if len(rows) == 0:
                continue
            sub_ctx = {k: v[rows] for k, v in ctx.items()}
            u, _, count_lp, eta = self.terms(sub_ctx, m)
            pres = _sample_nonempty_bernoulli(eta, gen)
            sub_out = torch.zeros((len(rows), self.C, self.P), dtype=torch.float64)
            for c in range(self.C):
                act = (pres[:, c] > 0.5).nonzero(as_tuple=True)[0]
                if len(act) > 0:
                    sizes_c = (
                        torch.multinomial(
                            count_lp[act, c].exp(), 1, generator=gen
                        ).ravel()
                        + 1
                    )
                    sub_out[act, c] = _sample_esp_subset(u[act, c], sizes_c, gen)
            out[rows] = sub_out
        return out


class PairwiseBinaryAR(nn.Module):
    """Baseline 7: Personalized pairwise product autoregression (tril(W, -1)), exact nonempty conditioning."""

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {**config, "benchmark_baseline": "pairwise_binary_ar"}
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((int(config["users"]), self.J), dtype=torch.float64),
            requires_grad=(parent is None and price is None),
        )
        self.price = nn.Parameter(
            init_p, requires_grad=(parent is None and price is None)
        )
        self.W = nn.Parameter(torch.zeros((self.J, self.J), dtype=torch.float64))
        self.cross_left = nn.Parameter(
            torch.zeros((self.J, 4), dtype=torch.float64), requires_grad=False
        )
        self.cross_right = nn.Parameter(
            torch.zeros((self.J, 4), dtype=torch.float64), requires_grad=False
        )
        self._split_cache = {}

    def _base_u(self, ctx: dict) -> torch.Tensor:
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.J)
        if self.parent is not None:
            with torch.no_grad():
                base = (self.parent.lam + self.parent.house[ctx["house"]]).reshape(
                    B, self.J
                )
            return base - self.base_price * d
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        if not self.price.requires_grad and not self.house.requires_grad:
            return base - self.base_price * d
        return base + self.house[ctx["house"]] - self.price * d

    def unary(self, ctx: dict) -> torch.Tensor:
        return self._base_u(ctx) + self.lam

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if (
            not grad_in
            and not self.price.requires_grad
            and not self.house.requires_grad
        ):
            if cache_key not in self._split_cache:
                with torch.no_grad():
                    self._split_cache[cache_key] = (
                        self._base_u(ctx),
                        y.reshape(-1, self.J).double(),
                    )
            u0, y_flat = self._split_cache[cache_key]
            u = u0 + self.lam
            W_tril = torch.tril(self.W, -1)
            eta = u + y_flat @ W_tril.T
            return (
                y_flat * F.logsigmoid(eta) + (1.0 - y_flat) * F.logsigmoid(-eta)
            ).sum(-1) - _nonempty_ar_normalizer(u)

        y_flat = y.reshape(-1, self.J).double()
        u = self.unary(ctx)
        W_tril = torch.tril(self.W, -1)
        eta = u + y_flat @ W_tril.T
        return (y_flat * F.logsigmoid(eta) + (1.0 - y_flat) * F.logsigmoid(-eta)).sum(
            -1
        ) - _nonempty_ar_normalizer(u)

    @torch.no_grad()
    def exact_mass(self, ctx: dict, sets: torch.Tensor) -> torch.Tensor:
        y_sets = sets.reshape(-1, self.J).double()
        u_all = self.unary(ctx)
        norm_all = _nonempty_ar_normalizer(u_all)
        W_tril = torch.tril(self.W, -1)
        shift = y_sets @ W_tril.T
        out = torch.empty((len(u_all), len(y_sets)), dtype=torch.float64)
        for s in range(0, len(u_all), 64):
            e = min(s + 64, len(u_all))
            eta = u_all[s:e, None, :] + shift[None, :, :]
            ll = (
                y_sets[None, :, :] * F.logsigmoid(eta)
                + (1.0 - y_sets[None, :, :]) * F.logsigmoid(-eta)
            ).sum(-1) - norm_all[s:e, None]
            out[s:e] = ll.exp()
        return out

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        B = len(ctx["house"])
        u = self.unary(ctx)
        W_tril = torch.tril(self.W, -1)
        suffix = [u.new_full((B,), -torch.inf)]
        for j in range(self.J - 1, -1, -1):
            suffix.append(
                torch.logaddexp(
                    F.logsigmoid(u[:, j]), F.logsigmoid(-u[:, j]) + suffix[-1]
                )
            )
        suffix.reverse()
        y = torch.zeros_like(u)
        active = torch.zeros(B, dtype=torch.bool, device=u.device)
        gen = torch.Generator().manual_seed(seed)
        for j in range(self.J):
            eta_j = u[:, j] + (y[:, :j] * W_tril[j, :j]).sum(-1)
            q = torch.where(
                active,
                eta_j.sigmoid(),
                (F.logsigmoid(u[:, j]) - suffix[j]).exp(),
            ).clamp(0, 1)
            take = torch.rand(B, generator=gen, device=u.device, dtype=u.dtype) < q
            y[:, j] = take.to(u.dtype)
            active |= take
        return y.reshape(B, self.C, self.P)


class LowRankBilinearAR(nn.Module):
    """Baseline 8: Fixed-Order Low-Rank Bilinear Binary Autoregression.

    Models conditional logits in a fixed product order using rank-R user-item
    factors and rank-R prefix interaction factors tril(alpha @ rho^T, -1) plus
    a running count slope, with exact nonempty-basket conditioning.
    (Note: This is a fixed-order binary autoregression, not the unordered
    sequential choice-and-checkout process of Ruiz, Athey & Blei 2020.)
    """

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        rank: int = 8,
        seed: int = 0,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {
            **config,
            "benchmark_baseline": f"low_rank_bilinear_ar_rank{rank}",
        }
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.rank = rank
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        g = torch.Generator().manual_seed(seed)
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.user_theta = nn.Parameter(
            0.05
            * torch.randn(
                (int(config["users"]), rank), generator=g, dtype=torch.float64
            ),
            requires_grad=(parent is None and price is None),
        )
        self.item_alpha = nn.Parameter(
            0.15 * torch.randn((self.J, rank), generator=g, dtype=torch.float64),
            requires_grad=False,
        )
        self.item_rho = nn.Parameter(
            0.05 * torch.randn((self.J, rank), generator=g, dtype=torch.float64)
        )
        self.price = nn.Parameter(
            init_p, requires_grad=(parent is None and price is None)
        )
        self.size_slope = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self._split_cache = {}

    def _base_u(self, ctx: dict) -> torch.Tensor:
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.J)
        if self.parent is not None:
            with torch.no_grad():
                base = (self.parent.lam + self.parent.house[ctx["house"]]).reshape(
                    B, self.J
                )
            return base - self.base_price * d
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        if not self.price.requires_grad and not self.user_theta.requires_grad:
            return base - self.base_price * d
        return base + self.user_theta[ctx["house"]] @ self.item_alpha.T - self.price * d

    def unary(self, ctx: dict) -> torch.Tensor:
        return self._base_u(ctx) + self.lam

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if (
            not grad_in
            and not self.price.requires_grad
            and not self.user_theta.requires_grad
        ):
            if cache_key not in self._split_cache:
                with torch.no_grad():
                    u0 = self._base_u(ctx)
                    y_flat = y.reshape(-1, self.J).double()
                    counts_before = y_flat.cumsum(-1) - y_flat
                    self._split_cache[cache_key] = (u0, y_flat, counts_before)
            u0, y_flat, counts_before = self._split_cache[cache_key]
            u = u0 + self.lam
            W = torch.tril(self.item_alpha @ self.item_rho.T, -1)
            eta = u + y_flat @ W.T + counts_before * self.size_slope[None, :]
            return (
                y_flat * F.logsigmoid(eta) + (1.0 - y_flat) * F.logsigmoid(-eta)
            ).sum(-1) - _nonempty_ar_normalizer(u)

        y_flat = y.reshape(-1, self.J).double()
        u = self.unary(ctx)
        W = torch.tril(self.item_alpha @ self.item_rho.T, -1)
        counts_before = y_flat.cumsum(-1) - y_flat
        eta = u + y_flat @ W.T + counts_before * self.size_slope[None, :]
        return (y_flat * F.logsigmoid(eta) + (1.0 - y_flat) * F.logsigmoid(-eta)).sum(
            -1
        ) - _nonempty_ar_normalizer(u)

    @torch.no_grad()
    def exact_mass(self, ctx: dict, sets: torch.Tensor) -> torch.Tensor:
        y_sets = sets.reshape(-1, self.J).double()
        u_all = self.unary(ctx)
        norm_all = _nonempty_ar_normalizer(u_all)
        W = torch.tril(self.item_alpha @ self.item_rho.T, -1)
        counts_before = y_sets.cumsum(-1) - y_sets
        shift = y_sets @ W.T + counts_before * self.size_slope[None, :]
        out = torch.empty((len(u_all), len(y_sets)), dtype=torch.float64)
        for s in range(0, len(u_all), 64):
            e = min(s + 64, len(u_all))
            eta = u_all[s:e, None, :] + shift[None, :, :]
            ll = (
                y_sets[None, :, :] * F.logsigmoid(eta)
                + (1.0 - y_sets[None, :, :]) * F.logsigmoid(-eta)
            ).sum(-1) - norm_all[s:e, None]
            out[s:e] = ll.exp()
        return out

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        B = len(ctx["house"])
        u = self.unary(ctx)
        W = torch.tril(self.item_alpha @ self.item_rho.T, -1)
        suffix = [u.new_full((B,), -torch.inf)]
        for j in range(self.J - 1, -1, -1):
            suffix.append(
                torch.logaddexp(
                    F.logsigmoid(u[:, j]), F.logsigmoid(-u[:, j]) + suffix[-1]
                )
            )
        suffix.reverse()
        y = torch.zeros_like(u)
        count = torch.zeros(B, dtype=torch.float64, device=u.device)
        active = torch.zeros(B, dtype=torch.bool, device=u.device)
        gen = torch.Generator().manual_seed(seed)
        for j in range(self.J):
            eta_j = u[:, j] + (y[:, :j] * W[j, :j]).sum(-1) + count * self.size_slope[j]
            q = torch.where(
                active,
                eta_j.sigmoid(),
                (F.logsigmoid(u[:, j]) - suffix[j]).exp(),
            ).clamp(0, 1)
            take = torch.rand(B, generator=gen, device=u.device, dtype=u.dtype) < q
            y[:, j] = take.to(u.dtype)
            count += take.to(u.dtype)
            active |= take
        return y.reshape(B, self.C, self.P)


class GaussianFactorMixture(nn.Module):
    """Baseline 9: Finite Gaussian-Node Factor Mixture over Nonempty Baskets.

    Mixes M=12 fixed Gaussian latent trip factor states z_m in R^K with
    factor loadings V in R^{J x K}, household mixture weights, and exact
    nonempty-basket normalization per latent state.
    (Note: This is an exact-normalized finite Gaussian-node mixture, not an
    amortized variational autoencoder with an inference network as in Liang et al. 2018.)
    """

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        n_factors: int = 6,
        n_nodes: int = 16,
        seed: int = 0,
        parent: nn.Module | None = None,
    ):
        super().__init__()
        self.config = {
            **config,
            "benchmark_baseline": f"gaussian_factor_mixture_K{n_factors}_M{n_nodes}",
        }
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        self.n_factors = n_factors
        self.n_nodes = n_nodes
        self.parent = parent
        if self.parent is not None:
            for p in self.parent.parameters():
                p.requires_grad_(False)
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        self.register_buffer("base_price", init_p)
        g = torch.Generator().manual_seed(seed)
        z_nodes = torch.randn((n_nodes, n_factors), generator=g, dtype=torch.float64)
        z_nodes[0].zero_()
        self.register_buffer("z_nodes", z_nodes)
        self.lam = nn.Parameter(
            torch.zeros(self.J, dtype=torch.float64), requires_grad=(parent is not None)
        )
        self.price = nn.Parameter(init_p, requires_grad=False)
        self.V = nn.Parameter(
            0.35 * torch.randn((self.J, n_factors), generator=g, dtype=torch.float64),
            requires_grad=(parent is not None),
        )
        self.node_scale = nn.Parameter(
            torch.ones(n_nodes, dtype=torch.float64), requires_grad=False
        )
        self.house_w = nn.Parameter(
            torch.zeros((int(config["users"]), n_nodes), dtype=torch.float64)
        )
        self.node_bias = nn.Parameter(torch.zeros(n_nodes, dtype=torch.float64))
        self._split_cache = {}

    def _base_u(self, ctx: dict):
        B = len(ctx["house"])
        d = ctx["dlp"].reshape(B, self.J)
        if self.parent is not None:
            with torch.no_grad():
                u0 = self.parent.utilities(ctx).reshape(B, self.J)
            return u0 + self.lam
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        return base + self.lam - self.price * d

    def _base_and_shifts(self, ctx: dict):
        u_base = self._base_u(ctx)
        shifts = (self.z_nodes * self.node_scale[:, None]) @ self.V.T
        log_w = F.log_softmax(
            self.node_bias[None, :] + self.house_w[ctx["house"]], dim=-1
        )
        return u_base, shifts, log_w

    def node_utilities(self, ctx: dict):
        u_base, shifts, log_w = self._base_and_shifts(ctx)
        return u_base[:, None, :] + shifts[None, :, :], log_w

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        grad_in = any(
            isinstance(v, torch.Tensor) and v.requires_grad for v in ctx.values()
        )
        cache_key = (id(ctx["dlp"]), id(y))
        if (
            not grad_in
            and not self.V.requires_grad
            and not self.node_scale.requires_grad
        ):
            if cache_key not in self._split_cache:
                with torch.no_grad():
                    y_flat = y.reshape(-1, self.J).double()
                    u_base = self._base_u(ctx)
                    shifts = self.z_nodes @ self.V.T
                    u_m = u_base[:, None, :] + shifts[None, :, :]
                    log_all = F.softplus(u_m).sum(-1)
                    log_Z = log_all + torch.log(
                        (-torch.expm1(-log_all)).clamp(min=1e-30)
                    )
                    y_dot = (y_flat * u_base).sum(-1, keepdim=True) + (
                        y_flat @ self.V
                    ) @ self.z_nodes.T
                    self._split_cache[cache_key] = y_dot - log_Z
            comp_base = self._split_cache[cache_key]
            log_w = F.log_softmax(
                self.node_bias[None, :] + self.house_w[ctx["house"]], dim=-1
            )
            return torch.logsumexp(comp_base + log_w, dim=-1)

        y_flat = y.reshape(-1, self.J).double()
        u_base, shifts, log_w = self._base_and_shifts(ctx)
        u_m = u_base[:, None, :] + shifts[None, :, :]
        log_all = F.softplus(u_m).sum(-1)
        log_Z = log_all + torch.log((-torch.expm1(-log_all)).clamp(min=1e-30))
        y_dot = (y_flat * u_base).sum(-1, keepdim=True) + (y_flat @ self.V) @ (
            self.z_nodes * self.node_scale[:, None]
        ).T
        comp = y_dot - log_Z + log_w
        return torch.logsumexp(comp, dim=-1)

    @torch.no_grad()
    def exact_mass(self, ctx: dict, sets: torch.Tensor) -> torch.Tensor:
        y_sets = sets.reshape(-1, self.J).double()
        u_m, log_w = self.node_utilities(ctx)
        log_all = F.softplus(u_m).sum(-1)
        log_Z = log_all + torch.log((-torch.expm1(-log_all)).clamp(min=1e-30))
        out = torch.empty((len(u_m), len(y_sets)), dtype=torch.float64)
        for s in range(0, len(u_m), 64):
            e = min(s + 64, len(u_m))
            dot = u_m[s:e] @ y_sets.T
            comp = dot - log_Z[s:e, :, None] + log_w[s:e, :, None]
            out[s:e] = torch.logsumexp(comp, dim=1).exp()
        return out

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        u_m, log_w = self.node_utilities(ctx)
        gen = torch.Generator().manual_seed(seed)
        B = len(ctx["house"])
        chosen_m = torch.multinomial(log_w.exp(), 1, generator=gen).ravel()
        u_sel = u_m[torch.arange(B, device=u_m.device), chosen_m]
        return _sample_nonempty_bernoulli(u_sel, gen).reshape(B, self.C, self.P)


# Backward-compatible aliases
ShopperFactorAR = LowRankBilinearAR
LogisticNormalFactorMixture = GaussianFactorMixture


class FastNeuralNADE(nn.Module):
    """Baseline 10: Conditional Neural Autoregressive Distribution Estimator (NADE, hidden width 16, rank-8 cross-price)."""

    def __init__(
        self,
        config: dict,
        price: torch.Tensor | None = None,
        hidden: int = 16,
        seed: int = 972301,
    ):
        super().__init__()
        self.config = {
            **config,
            "benchmark_baseline": f"conditional_NADE_H{hidden}_crossprice_rank8",
        }
        self.C, self.P = int(config["C"]), int(config["P"])
        self.J = self.C * self.P
        init_p = (
            price.detach().clone()
            if price is not None
            else torch.ones(self.J, dtype=torch.float64)
        )
        g = torch.Generator().manual_seed(seed)
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((int(config["users"]), self.J), dtype=torch.float64)
        )
        self.price = nn.Parameter(init_p)
        self.E = nn.Parameter(
            0.05 * torch.randn((self.J, hidden), generator=g, dtype=torch.float64)
        )
        self.Q = nn.Parameter(
            0.05 * torch.randn((self.J, hidden), generator=g, dtype=torch.float64)
        )
        self.O = nn.Parameter(torch.zeros((self.J, hidden), dtype=torch.float64))
        self.cross_left = nn.Parameter(torch.zeros((self.J, 8), dtype=torch.float64))
        g2 = torch.Generator().manual_seed(seed)
        self.cross_right = nn.Parameter(
            0.01 * torch.randn((self.J, 8), generator=g2, dtype=torch.float64)
        )

    def _unary_and_dQ(self, ctx: dict) -> tuple[torch.Tensor, torch.Tensor]:
        B = len(ctx["house"])
        base = ctx["baseline"].reshape(B, self.J) if "baseline" in ctx else 0.0
        d = ctx["dlp"].reshape(B, self.J)
        u = (
            base
            + self.lam
            + self.house[ctx["house"]]
            - self.price * d
            + (d @ self.cross_right) @ self.cross_left.T
            - d * (self.cross_left * self.cross_right).sum(-1)
        )
        return u, d @ self.Q

    def logits(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        u, d_Q = self._unary_and_dQ(ctx)
        contrib = y[..., None] * self.E
        prefix = contrib.cumsum(1) - contrib
        return u + (torch.sigmoid(prefix + d_Q[:, None]) * self.O).sum(-1)

    def log_probability(self, ctx: dict, y: torch.Tensor) -> torch.Tensor:
        y_flat = y.reshape(-1, self.J).double()
        u, d_Q = self._unary_and_dQ(ctx)
        contrib = y_flat[..., None] * self.E
        prefix = contrib.cumsum(1) - contrib
        eta = u + (torch.sigmoid(prefix + d_Q[:, None]) * self.O).sum(-1)
        empty = u + torch.sigmoid(d_Q) @ self.O.T
        return (y_flat * F.logsigmoid(eta) + (1.0 - y_flat) * F.logsigmoid(-eta)).sum(
            -1
        ) - _nonempty_ar_normalizer(empty)

    @torch.no_grad()
    def sample(self, ctx: dict, seed: int = 0) -> torch.Tensor:
        batch = len(ctx["house"])
        base = ctx["baseline"].reshape(batch, self.J) if "baseline" in ctx else 0.0
        d = ctx["dlp"].reshape(batch, self.J)
        unary = (
            base
            + self.lam
            + self.house[ctx["house"]]
            - self.price * d
            + (d @ self.cross_right) @ self.cross_left.T
            - d * (self.cross_left * self.cross_right).sum(-1)
        )
        price_hidden = d @ self.Q
        empty = unary + (price_hidden.sigmoid()[:, None] * self.O).sum(-1)
        suffix = [empty.new_full((batch,), -torch.inf)]
        for j in range(self.J - 1, -1, -1):
            suffix.append(
                torch.logaddexp(
                    F.logsigmoid(empty[:, j]),
                    F.logsigmoid(-empty[:, j]) + suffix[-1],
                )
            )
        suffix.reverse()
        prefix = price_hidden.clone()
        output = unary.new_zeros((batch, self.J))
        active = torch.zeros(batch, dtype=torch.bool, device=unary.device)
        generator = torch.Generator().manual_seed(seed)
        for j in range(self.J):
            logit = unary[:, j] + (prefix.sigmoid() * self.O[j]).sum(-1)
            probability = torch.where(
                active,
                logit.sigmoid(),
                (F.logsigmoid(empty[:, j]) - suffix[j]).exp(),
            ).clamp(0, 1)
            take = (
                torch.rand(
                    batch, generator=generator, device=unary.device, dtype=unary.dtype
                )
                < probability
            )
            output[:, j] = take.to(unary.dtype)
            prefix += take[:, None].to(unary.dtype) * self.E[j]
            active |= take
        return output.reshape(batch, self.C, self.P)
