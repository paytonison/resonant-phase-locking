# rpl_demo.py (pseudocode-level reference)
# This is NOT a drop-in torch module; it's meant to illustrate the mechanics clearly.

import math
import numpy as np

def rpl_attention(x, theta, Wq, Wk, Wv, Wo, lambda_phase=0.5, kappa=0.1, omega_fn=None, softmax=None):
    # x: [T, d], theta: [T] (radians)
    q = x @ Wq; k = x @ Wk; v = x @ Wv
    d = q.shape[-1]
    base_logits = (q @ k.T) / math.sqrt(d)

    dtheta = theta[:, None] - theta[None, :]
    phase_term = lambda_phase * np.cos(dtheta)
    logits = base_logits + phase_term
    attn = softmax(logits, axis=-1)

    omega = np.zeros_like(theta) if omega_fn is None else omega_fn(x)
    coupling = (attn * np.sin(dtheta)).sum(axis=-1)
    theta_next = theta + omega + kappa * coupling

    gate = np.maximum(0.0, np.cos(dtheta))
    y = (attn[..., None] * v[None, ...] * gate[..., None]).sum(axis=1)
    return y @ Wo, theta_next