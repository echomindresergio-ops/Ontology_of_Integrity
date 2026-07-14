# Пакет даних для χ²-фіту та бутстреп-аналізу DM_excess(b)
Зібрано: 2026-07-06. Джерела: GitHub (liamconnor), arXiv, Nature Astronomy, ApJL.

## Що ВДАЛОСЯ знайти (файли в цій папці)

### 1. connor2025_frbsample_individual.csv — індивідуальні виміри (пріоритет №2 з вашого чекліста)
Джерело: https://github.com/liamconnor/frb_baryon_connor2024/blob/main/data/frbsample_connor0924.csv
Стаття: Connor et al. (2025), "A gas-rich cosmic web revealed by the partitioning of the missing baryons",
Nature Astronomy 9, 1226. DOI: 10.1038/s41550-025-02566-y (arXiv:2409.16952).
69 локалізованих FRB. Колонки:
- name, TNSname, mjd, snr_heim, dm_heim, dm_opt — ідентифікація і сирий DM
- dm_exgal — екстрагалактичний DM (пк/см³), тобто DM_obs − DM_MW(NE2001)
- redshift, redshift_type (spec/photo), ra, dec, secure_host, survey
- ne2001 — внесок Чумацького Шляху за NE2001
- baryon_sample — чи входить у "золоту" вибірку статті
Це саме той список sightlines (object_id, z, DM, σ через ne2001-модель), з якого можна
самостійно робити бутстреп по об'єктах.

### 2. pdm_connor_etal_2025.h5 — лікелігуд-сітки P(DM|z) (замінник бутстреп-реалізацій, пріоритет №7)
Те саме репо. Вміст (HDF5):
- prob_dmcos_z (250×300) — P(DM_cos | z) (тільки IGM + гало, без хоста)
- prob_dmex_z (250×300) — P(DM_ex | z) (космос + хост)
- DM (250), redshift (300) — осі сітки
Це офіційні fitted Macquart PDF з найкращими параметрами MCMC статті — прямий вхід для
χ²/likelihood-фіту без потреби у власних MCMC-ланцюгах.

### 3. hussaini2025_table1_frb_sample.csv — вибірка стекінг-аналізу
Стаття: Hussaini, Connor, Konietzka, Ravi, Faber, Sharma, Sherman (2025),
"A Correlation between FRB DM and Foreground Large-scale Structure",
ApJL 993, L27. DOI: 10.3847/2041-8213/ae0a49 (arXiv:2506.04186).
Транскрибована Table 1 зі статті: 61 FRB (name, ra, dec, z, DM_obs, DM_exgal, survey).
Саме на цій вибірці побудовано Figure 6 — stacked ΔDM_cos(b⊥).

### 4. DM_excess_binned_TEMPLATE.csv — порожній шаблон під вашу схему колонок.

## Що НЕ опубліковано (треба просити в авторів) — пріоритет №1, №4

**Бінована таблиця ΔDM_cos(b⊥) (Fig. 6, Hussaini et al. 2025) і jackknife-ковариаційна
матриця 16×16 НЕ викладені** ні на Zenodo, ні на GitHub, ні в supplementary. Перевірено:
arXiv HTML/PDF, сторінку ApJL, Caltech Authors, GitHub liamconnor. Аналогічно для
Connor & Ravi (2022, Nat. Astron. 6, 1035): у Data availability зазначено лише
CHIME/FRB Catalog 1 і GWGC, а "custom code available from the corresponding author upon request".

Контакти для запиту:
- Maryam Hussaini — maryam.hussaini@cfa.harvard.edu (вказано в ApJL-версії)
- Liam Connor (CfA/Harvard), Vikram Ravi (Caltech)

### Чернетка листа (EN)
Subject: Data request: binned stacked DM_excess(b) and jackknife covariance (Hussaini et al. 2025, ApJL 993 L27)

Dear Dr. Hussaini,
I am reproducing the stacking analysis from your ApJL paper "A Correlation between FRB
Dispersion Measure and Foreground Large-scale Structure" for a chi-square fit and bootstrap
uncertainty analysis. Would you be able to share (CSV/ASCII):
(1) the binned stacked ΔDM_cos vs. impact parameter values underlying Figure 6, including
bin edges, mean ΔDM per bin, per-bin uncertainties, and pair counts per bin (16 bins);
(2) the 16×16 jackknife covariance matrix C_jack used for the χ²=46.5 estimate;
(3) if available, the individual FRB–galaxy pair list (FRB id, galaxy id, b⊥, ΔDM_cos).
Thank you very much for your time.
Best regards, [ім'я, афіліація]

## Метадані бінування/нормалізації (пріоритет №3) — зі статті Hussaini et al. 2025

- ΔDM_cos = DM_obs − DM_MW(NE2001, mwprop.ne2001p, до 30 кпк) − ⟨DM_cos(z)⟩ − ⟨DM_host⟩/(1+z_s);
  ⟨DM_host⟩ = 150 пк/см³; f_d = 0.93; f_e = 0.88; космологія Planck 2018.
- НЕ нормовано на M* чи Rvir — сирий ΔDM_cos, усереднений по всіх парах FRB–галактика в анулюсі.
- 16 бінів impact parameter; сигнал до ~Мпк масштабів; максимум b⊥ = 50 Мпк.
- Cuts вибірки: Legacy Survey (photo-z Zhou et al. 2021), 0.01 < z_gal < 0.5, log M* ≥ 10,
  z_gal ≤ 0.8·z_s, z_s > 0.10, |b_gal| > 5°, >2.5° від краю footprint.
- Оцінка похибок: (а) scramble FRB-позицій, 500 реалізацій → p=0.004 (~2.65σ);
  (б) jackknife по FRB → C_jack, χ²=46.5/16 dof → p=8×10⁻⁵ (~3.8σ).
- Rvir (r200) з M* через Moster et al. (2010).

## Властивості хостів/галактик (пріоритет №5) — публічні каталоги

- Legacy Survey DR9 photo-z + stellar masses (Zhou et al. 2021): https://www.legacysurvey.org/dr9/files/
  (photo-z каталог: https://www.legacysurvey.org/dr9/files/#photometric-redshifts)
- WISE-PS1-STRM (Beck et al. 2022) — класифікація + photo-z по 3π стерадіан.
- CHIME/FRB Catalog 1 (для Connor & Ravi 2022): https://www.chime-frb.ca/catalog
- GWGC (каталог галактик <40 Мпк, Connor & Ravi 2022): VizieR, source=GWGC.

## Суміжні відкриті набори, корисні для крос-перевірки

- FLIMFLAM DR1 (Khrykin et al. 2024, ApJ 973, 151) — foreground mapping FRB sightlines;
  дані/код: https://github.com/FRBs/FRB (репозиторій FRB community, doi:10.5281/zenodo.7991632).
- Wu & McQuinn (2023, ApJ 945, 87) — стекінг CHIME; код моделі CGMBrush (arXiv:2207.05233).
- Konietzka et al. (2025, in prep.) — TNG300 ray-tracing DM-каталог (1000 sightlines),
  використаний у Fig. 5/7 Hussaini et al.; станом на зараз ще не опублікований — просити разом.
