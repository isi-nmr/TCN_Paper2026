import math

import torch
from torch import nn

# def customMSELoss(networkInput, predicted, measured, mask, alpha=0.9, *, weight=True):
#     w = networkInput[:, [1], :] if weight else torch.tensor(1)


#     return (alpha) * torch.mean(((predicted - measured) * mask * w) ** 2)


class SymGeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [batch, channels, ...]
        return torch.sign(x) * x * 0.5 * (1 + torch.erf((x) / math.sqrt(2)))


def cumulative_trapezoid(y, dx=1.0, dim=-1):
    trap = 0.5 * (y[..., 1:] + y[..., :-1])
    integ = torch.cumsum(trap * dx, dim=dim)

    # prepend zero so output length matches input
    pad_shape = list(integ.shape)
    pad_shape[dim] = 1
    zero = torch.zeros(pad_shape, dtype=y.dtype, device=y.device)
    return torch.cat([zero, integ], dim=dim)


def cumulative_simpson(y, dx=1.0, dim=-1):
    y = torch.movedim(y, dim, -1)
    out = torch.zeros_like(y)
    n = y.shape[-1]

    if n >= 2:
        out[..., 1] = 0.5 * (y[..., 0] + y[..., 1]) * dx

    for i in range(2, n):
        if i % 2 == 0:
            out[..., i] = out[..., i - 2] + (dx / 3.0) * (y[..., i - 2] + 4.0 * y[..., i - 1] + y[..., i])
        else:
            out[..., i] = out[..., i - 1] + 0.5 * (y[..., i - 1] + y[..., i]) * dx

    return torch.movedim(out, -1, dim)


def integrate_trajectory(y, dt, method="trapz", dim=-1):
    if method == "cumsum":
        return torch.cumsum(y, dim=dim) * dt
    if method == "trapz":
        return cumulative_trapezoid(y, dx=dt, dim=dim)
    if method == "simpson":
        return cumulative_simpson(y, dx=dt, dim=dim)

    raise ValueError(f"Unsupported differentiable integration method for training loss: {method}")


def weighted_rms(x, weights, eps=1e-8):
    denom = torch.sum(weights).clamp_min(eps)
    return torch.sqrt(torch.sum((x**2) * weights) / denom).clamp_min(eps)


def compute_loss_terms(input, x, y, y_traj, mask, *, weight=False, eps=1e-8, dt=2e-6, integral_method="trapz"):
    w = torch.abs(input[:, [1], :]) if weight else torch.ones_like(x)
    weights = mask * w

    # ----- pointwise loss -----
    diff2 = (x - y) ** 2
    weighted = diff2 * weights

    denom = torch.sum(weights) + eps
    mse_term = torch.sum(weighted) / denom

    traj_pred = integrate_trajectory(x, dt, method=integral_method, dim=-1)
    traj_scale = weighted_rms(y_traj, weights, eps=eps)
    traj_err = (traj_pred - y_traj) / traj_scale
    integral_term = torch.sum((traj_err**2) * weights) / denom

    return mse_term, integral_term


def customMSELoss(input, x, y, y_traj, mask, *, alpha=1, weight=False, eps=1e-8, dt=2e-6, integral_method="trapz"):
    mse_term, integral_term = compute_loss_terms(
        input,
        x,
        y,
        y_traj,
        mask,
        weight=weight,
        eps=eps,
        dt=dt,
        integral_method=integral_method,
    )

    if alpha >= 1:
        return mse_term

    if alpha <= 0:
        return integral_term

    return alpha * mse_term + (1 - alpha) * integral_term


def trajMSELoss(networkInput, predicted, measured, mask, *, weight=False):
    w = networkInput[:, [1], :] if weight else torch.tensor(1)
    trajPred = cumulative_trapezoid(predicted, 2e-6)
    return torch.mean(((trajPred - measured) * mask * w) ** 2)
