"""Approximation analytique par MOMENT MATCHING (Levy) du call/put panier ARITHMETIQUE.

Le panier arithmetique B(T) = (1/d) sum_i S_i(T) n'est pas log-normal : pas de forme fermee.
Levy (1992) l'approxime par une log-normale de MEMES deux premiers moments (sous la mesure
risque-neutre), ce qui donne un prix de type Black-76.

Moments (S_i(0) = s0_i, vol sigma_i, correlation rho_ij, taux r, maturite T) :
  M1 = E[B]  = e^{rT} * mean_i(s0_i)
  M2 = E[B^2]= e^{2rT}/d^2 * sum_ij s0_i s0_j exp(rho_ij sigma_i sigma_j T)
puis   s^2 = ln(M2 / M1^2)   (variance log-normale totale),  F = M1  (forward du panier)
  call = e^{-rT} [ F N(d1) - K N(d2) ],  d1 = (ln(F/K)+0.5 s^2)/s,  d2 = d1 - s.

C'est une APPROXIMATION (exacte seulement a la limite de faible vol / forte correlation),
utilisee ici comme benchmark analytique, en pendant du pricer analytique geometrique.
"""

from math import erf, sqrt
import numpy as np


def _Phi(x):
    x = np.asarray(x, float)
    vec = np.vectorize(erf)
    return 0.5 * (1.0 + vec(x / sqrt(2.0)))


def moment_matching_prices(spots, vols, corr, r, T, K, callput_flag=0):
    """Prix MM (Levy) pour un LOT de vecteurs de spots.

    spots : (n, d) ; vols : (d,) ; corr : (d, d). Renvoie (n,).
    callput_flag : 0 = call, 1 = put.
    """
    S = np.atleast_2d(np.asarray(spots, float))          # (n, d)
    sig = np.asarray(vols, float)
    d = S.shape[1]

    M1 = np.exp(r * T) * S.mean(axis=1)                  # (n,)
    A = np.exp((corr * np.outer(sig, sig)) * T)          # (d, d)
    quad = np.einsum("ni,ij,nj->n", S, A, S)             # sum_ij S_i S_j A_ij
    M2 = np.exp(2 * r * T) / d ** 2 * quad               # (n,)

    s2 = np.log(np.maximum(M2 / M1 ** 2, 1.0 + 1e-15))   # >= 0
    s = np.sqrt(s2)
    F = M1
    d1 = (np.log(F / K) + 0.5 * s2) / s
    d2 = d1 - s
    disc = np.exp(-r * T)
    if callput_flag == 0:
        px = disc * (F * _Phi(d1) - K * _Phi(d2))
    else:
        px = disc * (K * _Phi(-d2) - F * _Phi(-d1))
    return px


def moment_matching_basket_price(derivative, market, tau=None):
    """Prix MM (Levy) au spot du `market`, signature type analytic_pricers (un seul point)."""
    sub = market.sub(derivative.underlying_ids)
    T = derivative.maturity if tau is None else float(tau)
    px = moment_matching_prices(sub.spots.reshape(1, -1), sub.vols, sub.corr_matrix,
                                sub.rate, T, derivative.strike, derivative.callput_flag)
    return float(px[0])