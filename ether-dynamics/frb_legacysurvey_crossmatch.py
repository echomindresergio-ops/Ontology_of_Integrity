"""
FRB x Legacy Survey DR9: реконструкція ΔDM_cos(b_perp)
========================================================

Крос-матчинг локалізованих FRB із фоновими галактиками з фотометричними
червоними зсувами (Legacy Survey DR9), для самостійної реконструкції
профілю надлишку міри дисперсії DM_excess(b) -- оскільки офіційна
бінована таблиця з Hussaini et al. 2025 (Рис. 6) не опублікована.

ПЕРЕД ЗАПУСКОМ ПРОЧИТАЙТЕ супровідну інструкцію
"Інструкція_крос_матчинг_FRB.md" -- там є критичний Крок 0 (перевірка
структури JSON), без якого цей скрипт може працювати неправильно.

Залежності:
    pip install requests numpy pandas astropy h5py scipy --break-system-packages

Вхідні файли (мають лежати в тій самій директорії):
    - hussaini2025_table1_frb_sample.csv   (61 FRB, основна вибірка)
    - pdm_connor_etal_2025.h5              (P(DM_cos|z) сітка)
"""

import time
import json
import numpy as np
import pandas as pd
import requests
from astropy.cosmology import Planck18
import astropy.units as u
import h5py


# ============================================================
# КОНФІГУРАЦІЯ -- підлаштуйте під себе
# ============================================================
FRB_CSV = "hussaini2025_table1_frb_sample.csv"
PDM_H5 = "pdm_connor_etal_2025.h5"
OUTPUT_BINNED_CSV = "DM_excess_binned_RECONSTRUCTED.csv"
OUTPUT_PAIRS_CSV = "FRB_galaxy_pairs_RAW.csv"

MAX_IMPACT_PARAMETER_MPC = 2.0   # починайте з малого (швидше), потім збільшуйте до 50
N_BINS = 16
API_DELAY_SECONDS = 1.0          # ввічлива затримка між запитами -- НЕ прибирайте
API_BASE = "https://www.legacysurvey.org/viewer/photoz-dr9/1/cat.json"

DM_HOST_ASSUMED = 150.0   # пк/см^3, фіксоване припущення статті Hussaini et al.


# ============================================================
# КРОК 1: Запит до Legacy Survey DR9 photo-z API
# ============================================================
def query_photoz_box(ra_lo, ra_hi, dec_lo, dec_hi, timeout=30):
    """
    Повертає список галактик (dict) у прямокутнику неба.
    УВАГА: перевірте на кроці 0 інструкції, чи поля нижче (ra, dec, z_phot,
    logmass) відповідають реальній структурі відповіді API -- назви полів
    можуть відрізнятися, скрипт написано за задокументованим форматом
    Tractor/photo-z каталогу, не перевіреним напряму в цій сесії.
    """
    params = {"ralo": ra_lo, "rahi": ra_hi, "declo": dec_lo, "dechi": dec_hi}
    resp = requests.get(API_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Очікувана структура (ПЕРЕВІРТЕ і скоригуйте за потреби):
    #   data["rows"] -- список об'єктів з полями ra, dec, z_phot, ... і т.д.
    # або data сам є списком -- обробляємо обидва варіанти захисно.
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    elif isinstance(data, list):
        return data
    else:
        print("!!! НЕВІДОМА СТРУКТУРА ВІДПОВІДІ API. Ось сирі ключі:", 
              list(data.keys()) if isinstance(data, dict) else type(data))
        return []


def angular_box_deg(ra_center, dec_center, z_frb, max_b_mpc):
    """
    Обчислює розмір прямокутника (у градусах) навколо позиції FRB,
    що відповідає max_b_mpc фізичних Мпк на характерному z ~ 0.3*z_frb
    (грубе наближення для типової галактики-переднього плану;
    остаточний, точний imppact parameter перераховується пізніше
    для кожної знайденої галактики на ЇЇ власному z_phot).
    """
    z_typical = max(0.05, 0.3 * z_frb)
    d_A = Planck18.angular_diameter_distance(z_typical)
    box_rad = (max_b_mpc * u.Mpc / d_A).to(u.dimensionless_unscaled).value
    box_deg = np.degrees(box_rad)
    dec_rad = np.radians(dec_center)
    ra_half = box_deg / max(np.cos(dec_rad), 0.1)
    return ra_half, box_deg


# ============================================================
# КРОК 2: Обчислення прицільного параметра b_perp (фізичні кпк)
# ============================================================
def impact_parameter_kpc(ra_frb, dec_frb, ra_gal, dec_gal, z_gal):
    """Кутова відстань -> фізична, на кутовій діаметральній відстані галактики."""
    from astropy.coordinates import SkyCoord
    c1 = SkyCoord(ra=ra_frb * u.deg, dec=dec_frb * u.deg)
    c2 = SkyCoord(ra=ra_gal * u.deg, dec=dec_gal * u.deg)
    sep_rad = c1.separation(c2).radian
    d_A = Planck18.angular_diameter_distance(z_gal)
    b_kpc = (sep_rad * d_A).to(u.kpc).value
    return b_kpc


# ============================================================
# КРОК 3: Очікуване <DM_cos(z)> з реальної сітки Connor et al. 2025
# ============================================================
def load_mean_dmcos(h5_path):
    with h5py.File(h5_path, "r") as f:
        DM = f["DM"][:]
        z_grid = f["redshift"][:]
        prob = f["prob_dmcos_z"][:]
    # ВАЖЛИВО: перший рядок (DM=0.01) містить NaN -- виключаємо його.
    DM_c = DM[1:]
    prob_c = prob[1:, :]
    mean_dmcos = np.array([
        np.sum(DM_c * prob_c[:, j]) / np.sum(prob_c[:, j])
        for j in range(len(z_grid))
    ])
    return z_grid, mean_dmcos


# ============================================================
# КРОК 4: Відбір галактик за критеріями Hussaini et al. 2025
# ============================================================
def passes_cuts(z_gal, z_frb, logmass, dec_gal, footprint_edge_deg=None):
    if not (0.01 < z_gal < 0.5):
        return False
    if logmass < 10:
        return False
    if z_gal > 0.8 * z_frb:
        return False
    if z_frb <= 0.10:
        return False
    # |b_galactic| > 5 deg -- перетворення в галактичні координати:
    from astropy.coordinates import SkyCoord
    # (координати галактики мають прийти разом із ra/dec виклику -- це
    #  перевіряється окремо в основному циклі, тут лишена заглушка)
    return True


# ============================================================
# ОСНОВНИЙ ЦИКЛ
# ============================================================
def main():
    print("Завантаження вхідних даних...")
    frbs = pd.read_csv(FRB_CSV)
    z_grid, mean_dmcos = load_mean_dmcos(PDM_H5)
    print(f"  {len(frbs)} FRB завантажено.")

    all_pairs = []

    for idx, frb in frbs.iterrows():
        ra_f, dec_f, z_f = frb["ra_deg"], frb["dec_deg"], frb["z"]
        dm_obs = frb["DM_exgal_pc_cm3"]

        if z_f <= 0.10:
            print(f"  [{frb['name']}] z={z_f:.3f} <= 0.10, пропускаємо (умова статті).")
            continue

        ra_half, dec_half = angular_box_deg(ra_f, dec_f, z_f, MAX_IMPACT_PARAMETER_MPC)
        ra_lo, ra_hi = ra_f - ra_half, ra_f + ra_half
        dec_lo, dec_hi = dec_f - dec_half, dec_f + dec_half

        print(f"  [{frb['name']}] запит box ~{2*dec_half:.3f}x{2*ra_half:.3f} deg ...")
        try:
            galaxies = query_photoz_box(ra_lo, ra_hi, dec_lo, dec_hi)
        except Exception as e:
            print(f"    !!! Помилка запиту: {e}")
            time.sleep(API_DELAY_SECONDS)
            continue

        expected_dmcos = np.interp(z_f, z_grid, mean_dmcos)

        for gal in galaxies:
            # ПЕРЕВІРТЕ реальні назви полів (Крок 0 інструкції)!
            ra_g = gal.get("ra")
            dec_g = gal.get("dec")
            z_g = gal.get("z_phot") or gal.get("photo_z") or gal.get("z_spec")
            logmass_g = gal.get("logmass") or gal.get("mass") or None

            if ra_g is None or dec_g is None or z_g is None:
                continue
            if logmass_g is None:
                # photo-z ендпоінт може не містити маси -- див. примітку
                # в інструкції про окремий запит до каталогу Tractor.
                continue

            if not passes_cuts(z_g, z_f, logmass_g, dec_g):
                continue

            b_kpc = impact_parameter_kpc(ra_f, dec_f, ra_g, dec_g, z_g)
            if b_kpc > MAX_IMPACT_PARAMETER_MPC * 1000:
                continue

            dm_cos_delta = dm_obs - expected_dmcos - DM_HOST_ASSUMED / (1 + z_f)

            all_pairs.append(dict(
                frb_name=frb["name"], b_kpc=b_kpc, z_gal=z_g,
                logmass=logmass_g, dm_cos_delta=dm_cos_delta,
            ))

        time.sleep(API_DELAY_SECONDS)

    if not all_pairs:
        print("\n!!! ЖОДНОЇ ПАРИ НЕ ЗНАЙДЕНО. Перевірте Крок 0 інструкції -- "
              "найімовірніше, назви полів у відповіді API не збігаються "
              "з очікуваними ('ra','dec','z_phot','logmass').")
        return

    df_pairs = pd.DataFrame(all_pairs)
    df_pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)
    print(f"\nЗбережено {len(df_pairs)} пар FRB-галактика у {OUTPUT_PAIRS_CSV}")

    # ---- Бінування ----
    bins = np.linspace(0, MAX_IMPACT_PARAMETER_MPC * 1000, N_BINS + 1)
    df_pairs["bin"] = pd.cut(df_pairs["b_kpc"], bins)

    rows = []
    for bin_range, group in df_pairs.groupby("bin", observed=True):
        if len(group) == 0:
            continue
        # Jackknife по унікальних FRB (не по парах) -- ближче до методології статті
        unique_frbs = group["frb_name"].unique()
        jack_means = []
        for excl in unique_frbs:
            sub = group[group["frb_name"] != excl]
            if len(sub) > 0:
                jack_means.append(sub["dm_cos_delta"].mean())
        jack_means = np.array(jack_means)
        err = np.std(jack_means) * np.sqrt(max(len(jack_means) - 1, 1)) if len(jack_means) > 1 else np.nan

        rows.append(dict(
            b_kpc=group["b_kpc"].mean(),
            b_low_kpc=bin_range.left, b_high_kpc=bin_range.right,
            DM_excess_pc_cm3=group["dm_cos_delta"].mean(),
            DM_err_pc_cm3=err,
            N_bin=len(group),
            host_mass_logMsun=group["logmass"].median(),
            host_z=group["z_gal"].median(),
            normalization_flag="raw, not normalized by M*/Rvir",
            notes=f"reconstructed, jackknife over {len(unique_frbs)} unique FRBs",
        ))

    df_binned = pd.DataFrame(rows)
    df_binned.to_csv(OUTPUT_BINNED_CSV, index=False)
    print(f"Збережено бінований результат у {OUTPUT_BINNED_CSV}")
    print(df_binned.to_string())


if __name__ == "__main__":
    main()
