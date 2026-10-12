""" 

"Erreur & speed-up d'un proxy de Tchebychev vs Monte Carlo" :

  - le proxy est construit UNIQUEMENT par Monte Carlo (chaque noeud de Tchebychev = un prix MC) ;
  - regime CRN / noCRN applique AUX DEUX endroits : noeuds (build) ET points de reference ;
  - métrique = erreur de PRIX (abs & rel) sur un echantillon de points du domaine,
    + mesures de temps (t_build, t_eval, t_mc_direct) et speed-ups (online / total).


"""

import math
import numpy as np

from monte_carlo import _correlated_normals
from monitoring import chrono




# =====================================================================
#  Domaine d'approximation (boite vol-scaled) & points de test
# =====================================================================
def vol_scaled_box(spots, vols, h_dom, conf_dom):
    """Boite [lo_i, hi_i] = S0_i * exp(+/- z * sigma_i * sqrt(h_dom)), z = Phi^-1(conf_dom).

    h_dom ne sert QU'A dimensionner la boite (le repricing reste a maturite T, h=0).
    """
    from math import sqrt
    # z = quantile normal de niveau conf_dom (sans scipy)
    z = _norm_ppf(conf_dom)
    half = z * np.asarray(vols) * sqrt(h_dom)
    lo = np.asarray(spots) * np.exp(-half)
    hi = np.asarray(spots) * np.exp(+half)
    return np.stack([lo, hi], axis=1)          # (d, 2)


def sample_box_uniform(box, n, seed):
    """`n` points tires uniformement dans la boite (n, d)."""
    rng = np.random.default_rng(seed)
    lo, hi = box[:, 0], box[:, 1]
    return lo + (hi - lo) * rng.random((n, len(lo)))


def _norm_ppf(p):
    """Quantile de la loi normale standard (Acklam), sans dependance externe."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# =====================================================================
#  Pricer MC aux noeuds : CRN (Z partage) / noCRN (Z frais a chaque appel)
# =====================================================================
class MCNodePricer(object):
    """Callable (point)->prix MC, signature MoCaX. Reutilise `derivative.payoff`.

    CRN   : un seul tirage Z (M_paths, d) reutilise a TOUS les appels : (AI CLAIM) proxy lisse, build rapide).
    noCRN : un Z frais a chaque appel (le generateur avance) : (AI CLAIM) bruit MC different partout.
    """
    __slots__ = ("deriv", "sub", "r", "tau", "vol", "sqrt_tau", "drift",
                 "M", "regime", "_Z", "_rng", "d", "n_calls")

    def __init__(self, derivative, market, tau, M_paths, regime, seed):
        self.deriv = derivative
        self.sub = market.sub(derivative.underlying_ids)
        self.r = self.sub.rate
        self.tau = float(tau)
        self.vol = self.sub.vols
        self.d = len(self.vol)
        self.sqrt_tau = math.sqrt(self.tau)
        self.drift = (self.r - 0.5 * self.vol ** 2) * self.tau
        self.M = int(M_paths)
        self.regime = regime
        self.n_calls = 0
        if regime == "CRN":
            rng = np.random.default_rng(seed)
            self._Z = _correlated_normals(rng, self.sub.corr_matrix, self.M)   # (M, d) fige
            self._rng = None
        elif regime == "noCRN":
            self._Z = None
            self._rng = np.random.default_rng(seed)                            # avance a chaque appel
        else:
            raise ValueError("regime inconnu: %r" % regime)

    def _draw(self):
        if self._Z is not None:
            return self._Z
        return _correlated_normals(self._rng, self.sub.corr_matrix, self.M)

    def __call__(self, point, additional_data=None):
        self.n_calls += 1
        s0 = np.asarray([float(point[i]) for i in range(self.d)])
        Z = self._draw()
        terminal = s0 * np.exp(self.drift + self.vol * self.sqrt_tau * Z)       # (M, d)
        payoff = self.deriv.payoff(terminal)
        return float(np.exp(-self.r * self.tau) * payoff.mean())

    def price_many(self, points):
        """Prix MC pour un lot de points (n, d) -> (n,)."""
        return np.array([self(p) for p in np.asarray(points)])



# =====================================================================
#  Erreurs
# =====================================================================
def price_errors(p_surr, p_ref):
    """abs & rel (max et moyenne) de p_surr vs p_ref, sur un vecteur de points."""
    p_surr, p_ref = np.asarray(p_surr, float), np.asarray(p_ref, float)
    eabs = np.abs(p_surr - p_ref)
    denom = np.maximum(np.abs(p_ref), 1e-12)
    erel = eabs / denom
    return dict(eabs_max=float(eabs.max()), eabs_mean=float(eabs.mean()),
                erel_max=float(erel.max()), erel_mean=float(erel.mean()))