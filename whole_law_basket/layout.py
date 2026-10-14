"""One active-product layout for packed/padded baskets and unequal categories."""

import numbers
import numpy as np
import torch


class CategoryLayout:
    """Category-major active products; padding is never part of basket support."""

    def __init__(self, config):
        raw = next(
            (
                config[key]
                for key in ("category_sizes", "P_c", "P_list")
                if key in config and config[key] is not None
            ),
            None,
        )
        if raw is None:
            raw = [config["P"]] * config["C"]
        sizes = list(raw)
        if not sizes or any(
            isinstance(p, bool) or not isinstance(p, numbers.Integral) or p < 1
            for p in sizes
        ):
            raise ValueError(
                "category_sizes must be a nonempty list of positive integers"
            )
        self.sizes = tuple(int(p) for p in sizes)
        self.C, self.P, self.J = len(sizes), max(sizes), sum(sizes)
        if "C" in config and int(config["C"]) != self.C:
            raise ValueError("C disagrees with category_sizes")
        if "P" in config and int(config["P"]) != self.P:
            raise ValueError("P must equal max(category_sizes)")
        self.ragged = any(p != self.P for p in self.sizes)
        self.offsets = np.cumsum([0, *self.sizes]).tolist()
        self.cat_of = [c for c, p in enumerate(self.sizes) for _ in range(p)]
        self.valid_mask = (
            torch.arange(self.P)[None, :] < torch.tensor(self.sizes)[:, None]
        )
        self.indices = self.valid_mask.flatten().nonzero(as_tuple=True)[0]

    def config(self, config):
        return {**config, "C": self.C, "P": self.P, "category_sizes": list(self.sizes)}

    def flatten(self, value, finite=False):
        """Select real entries before checking finiteness; ignore arbitrary padding."""
        if value.ndim == 3 and tuple(value.shape[1:]) == (self.C, self.P):
            out = value.reshape(len(value), -1)[:, self.indices.to(value.device)]
        elif value.ndim == 2 and value.shape[1] == self.J:
            out = value
        elif value.ndim == 2 and value.shape[1] == self.C * self.P:
            out = value[:, self.indices.to(value.device)]
        else:
            raise ValueError(
                f"Expected [batch,{self.J}] or [batch,{self.C},{self.P}], got {tuple(value.shape)}"
            )
        if finite and not torch.isfinite(out).all():
            raise ValueError("Active product features must be finite")
        return out

    def padded(self, value, finite=False):
        flat = self.flatten(value, finite=finite)
        if not self.ragged:
            return flat.reshape(len(flat), self.C, self.P)
        out = flat.new_zeros((len(flat), self.C * self.P))
        out[:, self.indices.to(flat.device)] = flat
        return out.reshape(len(flat), self.C, self.P)

    def format(self, flat, reference):
        if self.ragged and reference.ndim == 2 and reference.shape[1] == self.J:
            return flat
        return self.padded(flat)

    def valid_baskets(self, baskets):
        flat = self.flatten(baskets)
        valid = (
            torch.isfinite(flat).all(-1)
            & ((flat == 0) | (flat == 1)).all(-1)
            & (flat.sum(-1) > 0)
        )
        if self.ragged and baskets.numel() == len(baskets) * self.C * self.P:
            padded = baskets.reshape(len(baskets), self.C, self.P)
            phantom = padded[:, ~self.valid_mask.to(baskets.device)]
            valid &= (phantom == 0).all(-1)
        return valid

    def safe_baskets(self, baskets):
        """Evaluate rejected rows at a valid placeholder, then mask their scores."""
        valid = self.valid_baskets(baskets)
        if valid.all():
            return baskets, valid
        flat = self.flatten(baskets)
        placeholder = torch.zeros_like(flat)
        placeholder[:, 0] = 1
        safe = torch.where(valid[:, None], flat, placeholder)
        return self.format(safe, baskets), valid
