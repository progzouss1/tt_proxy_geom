"""Pricer analytique pour options sur moyenne GEOMETRIQUE (reference de contrôle).

La moyenne géometrique de sous-jacents log-normaux correlés reste log-normale :
il existe donc une forme fermée de type Black-Scholes. On l'utilise UNIQUEMENT pour
valider le pricer Monte-Carlo (le pricing de production reste MC pour tous les types).

Soit d sous-jacents S_i, de vol annuelle sigma_i, correlation rho, taux r, maturite T.
  - indice géometrique  :     G(T) = (prod_i S_i(T))^(1/d)
  - variance annualisee :     sigma_G^2 = (1/d^2) * sum_ij sigma_i sigma_j rho_ij
  - forward de l'indice :      F = G0 * exp((r - 0.5*mean(sigma_i^2) + 0.5*sigma_G^2) T)
puis Black-76 sur (F, K, sigma_G, T).
"""

from math import erf, sqrt
import numpy as np


def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def analytic_geometric_price(derivative, market):
    """Prix exact d'une option sur moyenne geometrique (call/put), sous GBM correle."""
    sub = market.sub(derivative.underlying_ids)
    s0, vol, rho = sub.spots, sub.vols, sub.corr_matrix
    T, r, K = derivative.maturity, sub.rate, derivative.strike
    d = len(s0)

    sigG2 = float(vol @ rho @ vol) / d ** 2       # variance annualisee de l'indice
    sigG = np.sqrt(sigG2)
    G0 = np.exp(np.mean(np.log(s0)))              # moyenne geometrique des spots
    avg_var = np.mean(vol ** 2)
    F = G0 * np.exp((r - 0.5 * avg_var + 0.5 * sigG2) * T)   # forward de l'indice

    vT = sigG * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigG2 * T) / vT
    d2 = d1 - vT
    if derivative.callput_flag == 0:                 # 0 = call
        px = np.exp(-r * T) * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:                                            # 1 = put
        px = np.exp(-r * T) * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
    return float(px)