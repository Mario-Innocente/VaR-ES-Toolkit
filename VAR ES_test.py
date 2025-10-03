import numpy as np
from VAR_ES import (
    historical_var_es,
    normal_var_es_from_returns,
    student_t_var_es_mc_from_returns,
)

def test_es_ge_var_historical():
    """ES deve essere >= VaR sulla stessa distribuzione di perdite."""
    rng = np.random.default_rng(0)
    # serie di ritorni con code un po' più pesanti (mixture)
    R = np.r_[rng.normal(0.0003, 0.01, 120_000),
              rng.normal(-0.0002, 0.02, 40_000)]
    L = -R
    for a in (0.95, 0.99):
        var, es = historical_var_es(L, alpha=a)
        assert es + 1e-12 >= var  # tolleranza numerica minima

def test_normal_closed_form_matches_empirical():
    """Per ritorni Normali, VaR/ES analitici devono allinearsi agli empirici."""
    rng = np.random.default_rng(42)
    R = rng.normal(0.0005, 0.01, size=150_000)
    L = -R
    a = 0.99
    v_a, e_a, mu, sig = normal_var_es_from_returns(R, alpha=a)
    v_e = float(np.quantile(L, a))
    e_e = float(L[L >= v_e].mean())
    assert abs(v_e - v_a) < 1e-3
    assert abs(e_e - e_a) < 1e-3

def test_student_t_mc_reasonable(monkeypatch):
    """La versione Student-t (MC) deve produrre ES >= VaR e valori finiti."""
    rng = np.random.default_rng(7)
    R = rng.normal(0.0, 0.01, size=50_000)  # qualsiasi serie di ritorni
    a = 0.99
    v, e, mu, sig = student_t_var_es_mc_from_returns(R, df=7, alpha=a, n_sims=120_000, seed=11)
    assert np.isfinite(v) and np.isfinite(e)
    assert e + 1e-12 >= v
