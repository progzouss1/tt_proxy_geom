"""Modele de portefeuille : classes de base issues de l'ingestion du classeur.

Portefeuille artificiel : une unite de chaque produit (quantite = 1).
Les derives sont identifies par une cle artificielle (leur numero 0..499),
les sous-jacents par une cle artificielle (leur id 0..19).
"""

from dataclasses import dataclass
import numpy as np

OPTION_TYPES = {0: "geo", 1: "basket", 2: "BO", 3: "WO"}


@dataclass
class Underlying:
    """Un sous-jacent, identifie par une cle artificielle 0..19."""
    id: int
    spot: float          # S0
    vol: float           # volatilite annuelle


@dataclass
class Derivative:
    derivative_id: int
    option_type: int
    callput_flag: int
    strike: float
    maturity: float
    underlying_ids: tuple
    nominal: float            # quantite detenue (facteur d'echelle du prix)

    @property
    def dimension(self) -> int:
        """Nombre de sous-jacents = dimension du surrogate."""
        return len(self.underlying_ids)

    @property
    def type_name(self) -> str:
        return OPTION_TYPES[self.option_type]

    def payoff(self, terminal_spots):
        """Payoff a maturite, vectorise sur les trajectoires.

        Tous les produits ne dependent que des etats terminaux des sous-jacents.
        On agrege selon le type, puis on applique le call/put.

        terminal_spots : ndarray (n_paths, dimension), colonnes dans l'ordre
                         de `underlying_ids`.
        Retour         : ndarray (n_paths,).
        """
        S = np.asarray(terminal_spots, dtype=float)
        t = self.option_type
        if t == 0:                       # geometrique (moyenne geometrique)
            agg = np.exp(np.mean(np.log(S), axis=1))
        elif t == 1:                     # basket (moyenne arithmetique)
            agg = np.mean(S, axis=1)
        elif t == 2:                     # best-of (max)
            agg = np.max(S, axis=1)
        else:                            # worst-of (min)
            agg = np.min(S, axis=1)
        if self.callput_flag == 0:                       # 0 = call
            return np.maximum(agg - self.strike, 0.0)
        return np.maximum(self.strike - agg, 0.0)        # 1 = put


@dataclass
class MarketData:
    """Etat de marche. Seuls les spots varient d'un scenario a l'autre."""
    spots: np.ndarray          # (20,)
    vols: np.ndarray           # (20,)
    corr_matrix: np.ndarray    # (20, 20)
    rate: float

    def sub(self, ids):
        """Sous-marche restreint aux sous-jacents `ids` (pour pricer un derive)."""
        idx = list(ids)
        return MarketData(self.spots[idx].copy(), self.vols[idx].copy(),
                          self.corr_matrix[np.ix_(idx, idx)].copy(), self.rate)

    def with_spots(self, ids, new_spots):
        """Copie ou seuls les spots de `ids` sont remplaces (pour bâtir un surrogate)."""
        s = self.spots.copy()
        s[list(ids)] = np.asarray(new_spots, dtype=float)
        return MarketData(s, self.vols.copy(), self.corr_matrix.copy(), self.rate)


class Portfolio:
    """Conteneur iterable des derives, accessible par derivative_id (cloisonnement)."""

    def __init__(self, derivatives):
        # derivatives : dict[int, Derivative]
        self._derivatives = dict(derivatives)

    def __iter__(self):
        return iter(self._derivatives.values())

    def __getitem__(self, derivative_id):
        return self._derivatives[derivative_id]

    def __len__(self):
        return len(self._derivatives)

    @property
    def ids(self):
        return list(self._derivatives.keys())
