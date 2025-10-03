# var_es_toolkit.py
# VaR / Expected Shortfall (CVaR) Toolkit with interactive data intake.
# Features:
# - Accepts external data: paste tabular text, CSV, or clipboard (if pandas available).
# - At start: ask if user data represents losses, returns, or a mixed dataframe.
# - Always normalize to a consistent loss distribution L.
#
# Methods:
# - Historical Simulation
# - Parametric Normal (analytical, no SciPy needed)
# - Parametric Student-t (via Monte Carlo)
# - Monte Carlo multivariate portfolio simulation with Cholesky
#
# Dependencies: numpy (required), pandas (optional for clipboard/CSV).
# Author: Mario 🚀

import sys
import os
from math import log, sqrt, exp
import numpy as np

# ---------------------------------
# Optional pandas (clipboard/CSV IO)
# ---------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# =========================
# Inverse Normal (no SciPy)
# =========================
def norm_ppf(p):
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2*log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = sqrt(-2*log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

# =========================
# Core risk functions
# =========================
def historical_var_es(losses, alpha=0.95):
    L = np.asarray(losses, dtype=float)
    q = np.quantile(L, alpha)
    es = L[L >= q].mean()
    return float(max(q, 0.0)), float(max(es, 0.0))

def normal_var_es_from_returns(returns, alpha=0.95):
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sig = r.std(ddof=1)
    z = norm_ppf(alpha)
    var = -(mu - z*sig)  # VaR of L = -R
    phi = (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*z*z)
    es = (phi/(1-alpha))*sig - mu
    return float(max(var, 0.0)), float(max(es, 0.0)), mu, sig

def student_t_var_es_mc_from_returns(returns, df=7, alpha=0.95, n_sims=400_000, seed=11):
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sig = r.std(ddof=1)
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance.")
    scale = sig / np.sqrt(df/(df-2))
    samples = rng.standard_t(df, size=n_sims) * scale + mu
    L = -samples
    var = np.quantile(L, alpha)
    es = L[L >= var].mean()
    return float(max(var, 0.0)), float(max(es, 0.0)), mu, sig

def mc_portfolio_var_es_from_cov(mu_vec, cov_mat, weights, horizon_days=1, alpha=0.95, n_sims=400_000, seed=99):
    rng = np.random.default_rng(seed)
    mu_vec = np.asarray(mu_vec, dtype=float)
    cov_mat = np.asarray(cov_mat, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    mu_h = mu_vec * horizon_days
    cov_h = cov_mat * horizon_days
    Lch = np.linalg.cholesky(cov_h + 1e-12*np.eye(cov_h.shape[0]))
    Z = rng.standard_normal((cov_h.shape[0], n_sims))
    sims = (mu_h.reshape(-1,1) + Lch @ Z)
    r_p = (w @ sims).ravel()
    Lp = -r_p
    var = np.quantile(Lp, alpha)
    es = Lp[Lp >= var].mean()
    return float(max(var, 0.0)), float(max(es, 0.0))

# =========================
# Data intake / parsing
# =========================
SEP_HINT = """
You can paste:
- a single column or multi-column table (Excel, CSV text)
- supported delimiters: comma, semicolon, tab, space
Paste below, then press ENTER twice when done.
"""

def read_multiline_stdin():
    print(SEP_HINT)
    print("Paste data here:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        return None
    return text

def parse_tabular_text_to_array(text):
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        for sep in [",", ";", "\t"]:
            raw = raw.replace(sep, " ")
        parts = [p for p in raw.split() if p]
        row = []
        for p in parts:
            try:
                row.append(float(p))
            except ValueError:
                pass
        if row:
            rows.append(row)
    if not rows:
        return None
    maxc = max(len(r) for r in rows)
    data = np.full((len(rows), maxc), np.nan, dtype=float)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r
    keep = ~np.all(np.isnan(data), axis=0)
    data = data[:, keep]
    return data

def read_from_clipboard_or_csv():
    if not HAS_PANDAS:
        print("pandas not available: clipboard/CSV disabled.")
        return None
    print("Choose data source: [1] Clipboard  [2] CSV file path")
    mode = input("> ").strip()
    try:
        if mode == "1":
            df = pd.read_clipboard()
        elif mode == "2":
            path = input("CSV file path: ").strip().strip('"')
            df = pd.read_csv(path)
        else:
            return None
        arr = df.to_numpy(dtype=float)
        return arr
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def choose_column(arr, prompt="Column index (0-based): "):
    ncols = arr.shape[1]
    print(f"Available columns: 0..{ncols-1}")
    while True:
        idx = input(prompt).strip()
        try:
            j = int(idx)
            if 0 <= j < ncols:
                return j
        except:
            pass
        print("Invalid index, try again.")

def normalize_to_losses_interactive():
    print("\nWhat type of data do you have?")
    print("[1] Losses  [2] Returns  [3] Mixed (both returns + losses)")
    kind = input("> ").strip()

    print("\nHow do you want to provide the data?")
    print("[1] Paste here")
    print("[2] Clipboard / CSV (requires pandas)")
    print("[3] Skip (use demo portfolio)")
    src = input("> ").strip()

    arr = None
    if src == "1":
        text = read_multiline_stdin()
        if text:
            arr = parse_tabular_text_to_array(text)
    elif src == "2":
        arr = read_from_clipboard_or_csv()
    elif src == "3":
        return None, "demo"

    if arr is None:
        print("No valid data provided. Using demo.")
        return None, "demo"

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if kind == "1":  # losses
        j = 0 if arr.shape[1] == 1 else choose_column(arr, "Losses column index: ")
        L = arr[:, j].astype(float)
        if np.any(L < 0):
            print("Negative losses found: applying abs() for consistency.")
            L = np.abs(L)
        return L[~np.isnan(L)], "losses"

    elif kind == "2":  # returns
        j = 0 if arr.shape[1] == 1 else choose_column(arr, "Returns column index: ")
        R = arr[:, j].astype(float)
        R = R[~np.isnan(R)]
        L = -R
        return L, "returns"

    elif kind == "3":  # mixed
        print("Select columns (optional):")
        j_loss = input("Losses column index (empty for none): ").strip()
        j_ret  = input("Returns column index (empty for none): ").strip()
        L = None
        if j_loss != "":
            j = int(j_loss)
            L = arr[:, j].astype(float)
            if np.any(L < 0):
                print("Negative losses found: applying abs().")
                L = np.abs(L)
            L = L[~np.isnan(L)]
        if L is None and j_ret != "":
            j = int(j_ret)
            R = arr[:, j].astype(float)
            R = R[~np.isnan(R)]
            L = -R
        if L is None:
            print("No valid column selected: using first column as returns.")
            R = arr[:, 0].astype(float)
            R = R[~np.isnan(R)]
            L = -R
        return L, "mixed"

    else:
        print("Invalid data type choice. Using demo.")
        return None, "demo"

# =========================
# Demo/synthetic data
# =========================
def demo_portfolio_returns(T=750, seed=7):
    rng = np.random.default_rng(seed)
    N = 3
    mu = np.array([0.10, 0.08, 0.12]) / 252
    vol = np.array([0.25, 0.20, 0.30]) / np.sqrt(252)
    corr = np.array([[1.0, 0.6, 0.4],
                     [0.6, 1.0, 0.5],
                     [0.4, 0.5, 1.0]])
    cov = np.outer(vol, vol) * corr
    Lc = np.linalg.cholesky(cov + 1e-12*np.eye(N))
    Z = rng.standard_normal((T, N))
    R = mu + Z @ Lc.T
    w = np.array([0.4, 0.35, 0.25])
    r_p = R @ (w / w.sum())
    return r_p, R, w

# =========================
# Main CLI
# =========================
if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    print("=== VaR / ES Toolkit ===")
    try:
        alpha = float(input("Confidence level alpha [0.95]: ").strip() or "0.95")
    except:
        alpha = 0.95
    try:
        alpha2 = float(input("Second alpha [0.99]: ").strip() or "0.99")
    except:
        alpha2 = 0.99

    L, source_kind = normalize_to_losses_interactive()

    if source_kind != "demo":
        # Historical VaR/ES
        h_var1, h_es1 = historical_var_es(L, alpha=alpha)
        h_var2, h_es2 = historical_var_es(L, alpha=alpha2)

        print("\n=== Results (user data) ===")
        print(f"Historical  VaR {alpha:.2%}: {h_var1:.4%} | ES {alpha:.2%}: {h_es1:.4%}")
        print(f"Historical  VaR {alpha2:.2%}: {h_var2:.4%} | ES {alpha2:.2%}: {h_es2:.4%}")

        if source_kind in ("returns", "mixed"):
            R = -L
            n_var1, n_es1, mu, sig = normal_var_es_from_returns(R, alpha=alpha)
            n_var2, n_es2, _, _ = normal_var_es_from_returns(R, alpha=alpha2)
            t_var1, t_es1, _, _ = student_t_var_es_mc_from_returns(R, df=7, alpha=alpha, n_sims=300_000, seed=11)
            t_var2, t_es2, _, _ = student_t_var_es_mc_from_returns(R, df=7, alpha=alpha2, n_sims=300_000, seed=12)

            print("\nParametric Normal (from returns)")
            print(f"mu={mu:.6f}, sigma={sig:.6f}")
            print(f"VaR {alpha:.2%}: {n_var1:.4%} | ES {alpha:.2%}: {n_es1:.4%}")
            print(f"VaR {alpha2:.2%}: {n_var2:.4%} | ES {alpha2:.2%}: {n_es2:.4%}")

            print("\nParametric Student-t (MC, df=7)")
            print(f"VaR {alpha:.2%}: {t_var1:.4%} | ES {alpha:.2%}: {t_es1:.4%}")
            print(f"VaR {alpha2:.2%}: {t_var2:.4%} | ES {alpha2:.2%}: {t_es2:.4%}")

        else:
            print("\nNote: You provided a pure loss series.")
            print("Parametric models based on returns are skipped.")

    else:
        print("\n=== Demo mode: synthetic 3-asset portfolio ===")
        r_p, R_assets, w = demo_portfolio_returns(T=750, seed=7)

        h_var1, h_es1 = historical_var_es(-r_p, alpha=alpha)
        h_var2, h_es2 = historical_var_es(-r_p, alpha=alpha2)
        print("\nHistorical (portfolio)")
        print(f"VaR {alpha:.2%}: {h_var1:.4%} | ES {alpha:.2%}: {h_es1:.4%}")
        print(f"VaR {alpha2:.2%}: {h_var2:.4%} | ES {alpha2:.2%}: {h_es2:.4%}")

        n_var1, n_es1, mu, sig = normal_var_es_from_returns(r_p, alpha=alpha)
        n_var2, n_es2, _, _ = normal_var_es_from_returns(r_p, alpha=alpha2)
        t_var1, t_es1, _, _ = student_t_var_es_mc_from_returns(r_p, df=7, alpha=alpha, n_sims=300_000, seed=11)
        t_var2, t_es2, _, _ = student_t_var_es_mc_from_returns(r_p, df=7, alpha=alpha2, n_sims=300_000, seed=12)

        print("\nParametric Normal (portfolio returns)")
        print(f"mu={mu:.6f}, sigma={sig:.6f}")
        print(f"VaR {alpha:.2%}: {n_var1:.4%} | ES {alpha:.2%}: {n_es1:.4%}")
        print(f"VaR {alpha2:.2%}: {n_var2:.4%} | ES {alpha2:.2%}: {n_es2:.4%}")

        mu_vec = R_assets.mean(axis=0)
        cov_mat = np.cov(R_assets.T, ddof=1)
        mc_var1, mc_es1 = mc_portfolio_var_es_from_cov(mu_vec, cov_mat, w, horizon_days=1, alpha=alpha, n_sims=300_000, seed=99)
        mc_var2, mc_es2 = mc_portfolio_var_es_from_cov(mu_vec, cov_mat, w, horizon_days=1, alpha=alpha2, n_sims=300_000)

