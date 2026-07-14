# Лист-запит до д-р Maryam Hussaini

**Кому:** maryam.hussaini@cfa.harvard.edu
**Копія (за бажанням):** Liam Connor (CfA/Harvard), Vikram Ravi (Caltech) — співавтори статті
**Тема:** Data request: binned stacked DM_excess(b) and jackknife covariance (Hussaini et al. 2025, ApJL 993, L27)

---

Dear Dr. Hussaini,

I am reproducing and extending the stacking analysis from your paper "A Correlation between FRB Dispersion Measure and Foreground Large-scale Structure" (ApJL 993, L27; arXiv:2506.04186) for an independent chi-square fit and bootstrap uncertainty analysis, comparing your empirical DM_excess(b⊥) profile against an alternative theoretical model of circumgalactic gas distribution.

Would you be able to share (as CSV or plain ASCII):

1. **The binned stacked ΔDM_cos vs. impact-parameter values underlying Figure 6**, including bin edges, mean ΔDM per bin, per-bin uncertainties, and the number of FRB–galaxy pairs contributing to each of the 16 bins.

2. **The 16×16 jackknife covariance matrix** C_jack used to obtain your χ²=46.5 estimate, so that I can propagate the correct (correlated) uncertainties rather than assuming independent bins.

3. **If available, the individual FRB–galaxy pair list** (FRB identifier, galaxy identifier, impact parameter b⊥, and the corresponding ΔDM_cos contribution) underlying the stack — this would let me test whether specific alternative radial profiles are preferred without needing to re-run your full pipeline.

I fully understand if some of these products are not readily shareable in their current form, and I am happy to work with whatever subset is convenient for you to provide. If it is easier to point me to a code repository or a supplementary data release that already contains this, that would also be extremely helpful.

Thank you very much for your time, and congratulations on a very interesting result.

Best regards,
[Ваше ім'я]
[Ваша афіліація/незалежний дослідник]
[Контактна інформація]

---

## Практичні нотатки для вас (не для листа)

- Наведена вище чернетка — це доопрацьована версія тієї, що вже була у вашому README, з доданим явним поясненням мети (порівняння з альтернативною моделлю), що робить прохання конкретнішим і легше обґрунтованим для відповіді.
- Варто додати одне речення про те, хто ви є (незалежний дослідник, аматор-фізик, студент тощо) — авторам простіше відповісти, коли розуміють контекст запиту.
- Якщо за 2–3 тижні відповіді не буде, можна написати повторно й додати в копію Liam Connor — іноді співавтори відповідають швидше.
- Альтернатива, якщо відповіді не буде: самостійна реконструкція через крос-матчинг (див. окремий документ-завдання нижче) — вона не потребує дозволу авторів, оскільки використовує лише публічні каталоги (Legacy Survey DR9) та ваш власний список FRB.
