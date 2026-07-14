"""
Аналіз: порівняння моделей проти реальних крос-матчених даних FRB
====================================================================

Консолідований, придатний до повторного запуску скрипт, що відтворює
головний кількісний результат дослідження: порівняння ефірної моделі,
стандартної моделі густини гало (мНFW) і тривіальної "плоскої" моделі
проти реальних, крос-матчених даних FRB-galaxy (FRB_galaxy_pairs_RAW.csv).

Вхідні дані:
    FRB_galaxy_pairs_RAW.csv -- отримано скриптом frb_legacysurvey_crossmatch.py

Залежності:
    pip install numpy pandas scipy --break-system-packages
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad


# ============================================================
# Модельні криві для порівняння
# ============================================================

# Нормована форма DM(b) для ефірної моделі (з фінальної стабільної
# конфігурації, вітер v=300 км/с; див. ether_galaxy_model.py)
ETHER_MODEL_B_KPC = np.array([10, 30, 60, 100, 150, 200, 240, 280])
ETHER_MODEL_DM_NORM = np.array(
    [1.000, 1.030, 1.111, 1.206, 1.185, 0.883, 0.65, 0.55]
)


def rho_mnfw(r_kpc, r200_kpc=250.0, c=7.67, alpha=2.0, y0=2.0):
    """
    Модифікований профіль NFW (Prochaska & Zheng 2019), з ФІДУЦІАЛЬНИМ
    (не довільним!) значенням концентрації c=7.67 -- взято з реального
    коду пакету FRB (frb/halos/models.py), а не з приблизної формули.
    """
    y = np.maximum(c * (r_kpc / r200_kpc), 1e-8)
    return 1.0 / (y ** (1 - alpha) * (y0 + y) ** (2 + alpha))


def dm_line_of_sight(b_kpc, rho_func, r_max_kpc=250.0):
    """Інтеграл уздовж променя зору крізь сферичне гало, обрізане на r_max."""
    if b_kpc >= r_max_kpc:
        return 0.0
    integrand = lambda s: rho_func(np.sqrt(b_kpc**2 + s**2))
    val, _ = quad(integrand, 0, np.sqrt(max(r_max_kpc**2 - b_kpc**2, 0)), limit=100)
    return 2 * val


def mnfw_shape_at(b_values_kpc, r200_kpc=250.0, c=7.67):
    dms = np.array([
        dm_line_of_sight(b, lambda r: rho_mnfw(r, r200_kpc, c), r200_kpc)
        for b in b_values_kpc
    ])
    return dms / dms[0]


# ============================================================
# Бінування реальних даних (jackknife по унікальних FRB)
# ============================================================
def bin_pairs(df, b_max_kpc=300.0, n_bins=4, method="uniform"):
    """
    method: "uniform" (рівні за шириною біни) або "quantile"
    (рівна кількість пар у кожному біні -- ефективніше для малої вибірки).
    """
    close = df[df["b_kpc"] < b_max_kpc].copy()

    if method == "uniform":
        bins = np.linspace(0, b_max_kpc, n_bins + 1)
        close["bin"] = pd.cut(close["b_kpc"], bins)
    elif method == "quantile":
        close["bin"] = pd.qcut(close["b_kpc"], q=n_bins, duplicates="drop")
    else:
        raise ValueError("method має бути 'uniform' або 'quantile'")

    rows = []
    for _, group in close.groupby("bin", observed=True):
        if len(group) == 0:
            continue
        unique_frbs = group["frb_name"].unique()
        jack_means = []
        for excl in unique_frbs:
            sub = group[group["frb_name"] != excl]
            if len(sub) > 0:
                jack_means.append(sub["dm_cos_delta"].mean())
        jack_means = np.array(jack_means)
        n_j = len(jack_means)
        err = (np.std(jack_means) * np.sqrt(max(n_j - 1, 1))
               if n_j > 1 else np.nan)
        rows.append(dict(
            b_kpc=group["b_kpc"].mean(),
            DM_excess_pc_cm3=group["dm_cos_delta"].mean(),
            DM_err_pc_cm3=err,
            N_pairs=len(group),
            N_unique_frb=n_j,
        ))
    return pd.DataFrame(rows)


# ============================================================
# Зважений фіт масштабу (eta) і chi-square
# ============================================================
def fit_amplitude_and_chi2(data_dm, data_err, model_shape):
    """
    Підганяє єдиний масштабний параметр eta методом зважених найменших
    квадратів: minimize sum((data - eta*model)^2 / err^2).
    Повертає (eta, chi2, chi2/dof, прогнозовані значення).
    """
    weights = 1.0 / data_err**2
    eta = np.sum(weights * data_dm * model_shape) / np.sum(weights * model_shape**2)
    pred = eta * model_shape
    chi2 = np.sum((data_dm - pred)**2 / data_err**2)
    dof = len(data_dm) - 1  # -1 за підігнаний eta
    return eta, chi2, chi2 / dof, pred


def compare_models(binned_df, r200_kpc=250.0):
    """Тристороннє порівняння: ефірна модель / mNFW / плоска модель."""
    b = binned_df["b_kpc"].values
    data = binned_df["DM_excess_pc_cm3"].values
    err = binned_df["DM_err_pc_cm3"].values

    shapes = {
        "Ефірна (вітер 300 км/с)": np.interp(b, ETHER_MODEL_B_KPC, ETHER_MODEL_DM_NORM),
        "мНFW (c=7.67, коректне)": mnfw_shape_at(b, r200_kpc),
        "Плоска (стала, контроль)": np.ones_like(b),
    }

    print(f"{'Модель':>28} {'eta':>8} {'chi2':>8} {'chi2/dof':>10}")
    results = {}
    for name, shape in shapes.items():
        eta, chi2, chi2_dof, pred = fit_amplitude_and_chi2(data, err, shape)
        print(f"{name:>28} {eta:8.2f} {chi2:8.2f} {chi2_dof:10.2f}")
        results[name] = dict(eta=eta, chi2=chi2, chi2_dof=chi2_dof, pred=pred)
    return results


# ============================================================
# Додаткові дешеві перевірки (не потребують крос-матчингу)
# ============================================================
def test_mass_dependence(pairs_df, b_max_kpc=300.0, n_bins=4):
    """Чи залежить надлишок DM від маси гало-кандидата? (мало бути позитивним
    у будь-якій моделі, де ефір/газ пов'язаний з баріонною масою)."""
    close = pairs_df[pairs_df["b_kpc"] < b_max_kpc].copy()
    close["mass_bin"] = pd.qcut(close["logmass"], q=n_bins, duplicates="drop")
    rows = []
    for _, group in close.groupby("mass_bin", observed=True):
        rows.append((group["logmass"].mean(), group["dm_cos_delta"].mean()))
    logm, dm = zip(*rows)
    r, p = stats.pearsonr(logm, dm)
    print(f"Кореляція надлишку DM з масою гало-кандидата: r={r:.3f}, p={p:.3f}")
    return r, p


# ============================================================
# Точка входу
# ============================================================
def main():
    print("Завантаження реальних крос-матчених даних...")
    pairs = pd.read_csv("FRB_galaxy_pairs_RAW.csv")
    print(f"  {len(pairs)} пар, {pairs['frb_name'].nunique()} унікальних FRB\n")

    print("=== Рівномірні біни (4) ===")
    binned_uniform = bin_pairs(pairs, method="uniform", n_bins=4)
    print(binned_uniform.to_string(index=False))
    compare_models(binned_uniform)

    print("\n=== Квантильні біни (рівна кількість пар, 4) ===")
    binned_quantile = bin_pairs(pairs, method="quantile", n_bins=4)
    print(binned_quantile.to_string(index=False))
    compare_models(binned_quantile)

    print("\n=== Додаткова перевірка: залежність від маси гало ===")
    test_mass_dependence(pairs)


if __name__ == "__main__":
    main()
