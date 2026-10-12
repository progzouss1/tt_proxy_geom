"""Pricer Monte-Carlo generique + generation de scenarios 1 jour.

Tous les produits ne dependent que des etats terminaux des sous-jacents : on simule
S_T sous la mesure risque-neutre, on appelle `derivative.payoff`, on moyenne et on
actualise. Les tirages sont COMMUNS (meme seed re-appliquee a chaque appel de
`price`) : le prix devient une fonction LISSE des spots, condition necessaire a
l'interpolation de Tchebychev.

Le parametre `tau` permet de repricer a maturite RESIDUELLE (T - h) pour un calcul
de risque a horizon h : par defaut tau = derivative.maturity (valeur a t=0).
"""

import numpy as np


def _correlated_normals(rng, corr, n):
    """`n` tirages ~ N(0, corr), shape (n, d). Repli eigen si corr non-PSD."""
    d = corr.shape[0]
    z = rng.standard_normal((n, d))
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    return z @ L.T


class MonteCarloPricer:
    """Pricer generique par Monte-Carlo, a nombres aleatoires communs.

    price(derivative, market, tau=None) -> float
        tau : maturite residuelle a utiliser. None -> derivative.maturity (t=0).
    """

    def __init__(self, n_paths=20000, seed=12345):
        self.n_paths = int(n_paths)
        self.seed = int(seed)

    def price(self, derivative, market, tau=None):
        sub = market.sub(derivative.underlying_ids)
        T = derivative.maturity if tau is None else float(tau)   # maturite (residuelle si tau)
        r = sub.rate
        s0, vol = sub.spots, sub.vols

        # tirages communs : meme bruit a chaque appel -> prix lisse en s0
        rng = np.random.default_rng(self.seed)
        g = _correlated_normals(rng, sub.corr_matrix, self.n_paths)   # (n_paths, d)

        terminal = s0 * np.exp((r - 0.5 * vol ** 2) * T + vol * np.sqrt(T) * g)
        payoff = derivative.payoff(terminal)
        return float(np.exp(-r * T) * payoff.mean())