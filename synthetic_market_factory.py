"""Marche synthetique a d sous-jacents + fabrique de paniers (arithmetique / geometrique).

But : l'experience balaie d = 2..25, alors que le classeur reel n'a que 20 sous-jacents.
On genere donc un marche artificiel reproductible (seed fixe) dans les plages du papier
(spots 100-120, vols 0.2-0.5, r = 1 %) avec une correlation constante (defaut 40 %).

On NE touche PAS aux classes existantes : on reutilise `MarketData` et `Derivative`
de `portfolio_model`. Le `MonteCarloPricer` price ces objets via `derivative.payoff`
sans modification.

    market25 = make_synthetic_market(n=25, rho=0.40)          # MarketData (25 sous-jacents)
    deriv    = make_basket(d=8, market=market25, K=100, T=1.0) # call panier arithmetique 8D
    deriv_g  = make_basket(d=8, market=market25, K=100, T=1.0, geometric=True)  # geometrique
"""

import numpy as np

from model import MarketData, Derivative


def constant_corr(n, rho):
    """Matrice de correlation constante (1 sur la diag, rho ailleurs). PSD pour rho in [-1/(n-1), 1]."""
    C = np.full((n, n), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


def make_synthetic_market(n=25, rho=0.40, rate=0.01,
                          spot_range=(100.0, 120.0), vol_range=(0.20, 0.50),
                          seed=20240101):
    """Construit un `MarketData` de `n` sous-jacents correles a `rho` (constant).

    spots et vols sont tires uniformement dans les plages donnees, seed fixe -> reproductible.
    """
    rng = np.random.default_rng(seed)
    spots = rng.uniform(spot_range[0], spot_range[1], size=n)
    vols = rng.uniform(vol_range[0], vol_range[1], size=n)
    corr = constant_corr(n, rho)
    return MarketData(spots=spots, vols=vols, corr_matrix=corr, rate=float(rate))


def make_basket(d, market, K, T, geometric=False, callput_flag=0, nominal=1.0):
    """Fabrique un call/put panier sur les `d` premiers sous-jacents de `market`.

    geometric=False -> option_type=1 (moyenne ARITHMETIQUE)   [produit etudie]
    geometric=True  -> option_type=0 (moyenne GEOMETRIQUE)    [validation analytique]
    callput_flag : 0 = call (defaut), 1 = put.
    Renvoie un `Derivative` dont `underlying_ids = (0, 1, ..., d-1)`.
    """
    if d > len(market.spots):
        raise ValueError("d=%d > nombre de sous-jacents du marche (%d)" % (d, len(market.spots)))
    return Derivative(
        derivative_id=int(d),                 # cle artificielle = la dimension (1 derive par d)
        option_type=0 if geometric else 1,
        callput_flag=int(callput_flag),
        strike=float(K),
        maturity=float(T),
        underlying_ids=tuple(range(d)),
        nominal=float(nominal),
    )