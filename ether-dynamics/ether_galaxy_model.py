"""
Ефірна модель галактики — фінальна стабільна конфігурація
============================================================

Референсна реалізація 1D радіальної моделі "ефірного горизонту" галактики
з локальним порогом переходу, радіально залежним гістерезисом та
зовнішньою адвекцією ("ефірним вітром").

Ця конфігурація — результат довгого циклу діагностики й виправлень
(див. супровідний документ "Фінальна_конфігурація_детальний_опис.md").
Вона одночасно задовольняє три критерії, які окремо провалювали
попередні версії моделі:

    1. НЕ розганяється (немає каскадної нестабільності частоти подій).
    2. Зберігає ~97-99% енергії за 13.8 млрд років (вік Всесвіту).
    3. Дає периферійне домінування активності (80-100% подій при r>10 кпк).

Автор: підготовлено як робочий інструмент для подальшого дослідження.
Залежності: numpy, scipy (solve_ivp, quad).

------------------------------------------------------------------
ЯК ЦИМ КОРИСТУВАТИСЯ
------------------------------------------------------------------
    from ether_galaxy_model import EtherGalaxyModel

    model = EtherGalaxyModel(N=128, v_wind=300e3)
    result = model.run(event_cap=2000, time_budget_seconds=200)
    dm_b = model.dm_shape(result.u_e_avg_snapshot)

Дивіться функцію `_demo()` в кінці файлу для повного прикладу запуску,
включно з побудовою кривої DM(b) і перевіркою енергетичного бюджету.

------------------------------------------------------------------
ВІДОМІ ОБМЕЖЕННЯ (чесно, щоб не витрачати час на повторне відкриття)
------------------------------------------------------------------
    - N=256 обчислювально дуже дороге через густий якобіан BDF-розв'язувача
      SciPy (масштабується гірше за O(N^2)). Для реальної роботи на
      N>=256 бажана оптимізація: розріджений якобіан, Numba/JIT,
      або зовнішній розв'язувач (SUNDIALS/PETSc через petsc4py).
    - Переклад u_e (модельна "ефірна енергетична густина") у фізичну
      густину вільних електронів n_e для DM-обчислень — ЛІНІЙНИЙ І
      ДОВІЛЬНИЙ (n_e = eta * u_e). Немає незалежного фізичного
      обґрунтування коефіцієнта eta; будь-яке абсолютне (не лише
      формо-порівняльне) зіставлення з реальними DM-даними впирається
      саме в цей нерозв'язаний крок.
    - Параметр gamma (безперервна конденсація ефір->баріон) навмисно
      відсутній у цій реалізації: перевірений окремо, виявився або
      "мертвим", або другорядним модифікатором, що не змінює якісної
      картини. Дискретна перекачка xi — єдиний реальний канал конденсації.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
import time as _time
from dataclasses import dataclass, field


# ============================================================
# Фізичні та космологічні константи (SI, крім явно позначеного)
# ============================================================
PC = 3.0857e16       # м
KPC = 1000 * PC
MPC = 1000 * KPC
GYR = 3.156e16        # секунд в мільярді років


@dataclass
class SimulationResult:
    """Результат одного прогону моделі."""
    t_reached_s: float
    events_total: int
    energy_fraction_remaining: float
    frac_outer: float            # частка подій при r > 10 кпк
    u_e_avg_snapshot: np.ndarray  # усереднений по знімках профіль u_e(r)
    final_state: np.ndarray       # повний вектор стану [u_b, u_e, Psi]
    event_radii_kpc: list


class EtherGalaxyModel:
    """
    1D радіальна модель ефірної галактики з локальним Psi(r,t),
    радіально залежним гістерезисом та зовнішньою адвекцією.

    Усі параметри мають значення за замовчуванням, що відповідають
    фінальній стабільній конфігурації, описаній у супровідному документі.
    Змінюйте їх як конструктор-аргументи для експериментів.
    """

    def __init__(
        self,
        N: int = 128,
        r_min: float = 0.1 * PC,
        r_max: float = 3 * MPC,
        rs_baryon: float = 10 * KPC,
        kappa_flow: float = 1e-16,       # с^-1, інтенсивність прямого потоку
        Theta0: float = 1e-15,           # с^-1, швидкість накопичення напруги
        Lc: float = 3.0e-5,              # ВИПРАВЛЕНЕ значення втрати за подію
        xi: float = 0.1,                 # частка миттєвої перекачки ефір->баріон
        kappa_reset: float = 0.9,        # частка скидання Psi при події
        beta_on: float = 1.0,
        beta_c: float = 0.70,            # поріг деактивації в центрі
        beta_p: float = 0.98,            # поріг деактивації на периферії
        rs_hysteresis: float = 30 * KPC, # масштаб переходу гістерезису
        m_exponent: int = 1,
        Psi_c: float = 1.0,
        D: float = 1e25,                 # м^2/с, коефіцієнт дифузії
        v0_rot: float = 200e3,           # м/с, амплітуда базової швидкості
        r0_rot: float = 1 * KPC,
        v_wind: float = 300e3,           # м/с, стала зовнішня адвекція ("вітер")
        r200: float = 250 * KPC,         # умовний віріальний радіус
    ):
        self.N = N
        self.kappa_flow = kappa_flow
        self.Theta0 = Theta0
        self.Lc = Lc
        self.xi = xi
        self.kappa_reset = kappa_reset
        self.beta_on = beta_on
        self.m_exponent = m_exponent
        self.Psi_c = Psi_c
        self.D = D
        self.r200 = r200

        # --- Сітка (логарифмічна) ---
        self.r = np.geomspace(r_min, r_max, N)
        r_edges_inner = np.sqrt(self.r[:-1] * self.r[1:])
        self.r_edges = np.concatenate([
            [self.r[0]**2 / r_edges_inner[0]],
            r_edges_inner,
            [self.r[-1]**2 / r_edges_inner[-1]],
        ])
        self.dr = np.diff(self.r_edges)

        # --- Баріонний профіль (cored, слабко концентрований) ---
        rho_b = 1.0 / (1 + np.sqrt(self.r / rs_baryon)) ** 2
        self.rho_b_norm = rho_b / rho_b[0]   # нормовано: центр = 1

        # --- Швидкість переносу ефіру: базова "ротаційна" + вітер ---
        self.v_e = v0_rot * (self.r / r0_rot) ** (-0.5) + v_wind

        # --- Радіально залежний гістерезис ---
        f_r = 1 - 1 / (1 + self.r / rs_hysteresis)
        self.beta_off_r = beta_c + (beta_p - beta_c) * f_r

    # ------------------------------------------------------------------
    # Права частина ОДУ (транспорт + джерело), без дискретних подій
    # ------------------------------------------------------------------
    def _f_Psi(self, Psi):
        return np.clip(Psi / self.Psi_c, 0, None) ** self.m_exponent

    def _divergence_flux(self, u_e):
        u_edge = np.interp(self.r_edges[1:-1], self.r, u_e)
        v_edge = np.interp(self.r_edges[1:-1], self.r, self.v_e)
        dudr_edge = np.diff(u_e) / np.diff(self.r)
        J = u_edge * v_edge - self.D * dudr_edge
        # Гран. умови: J=0 у центрі (симетрія), відтік на зовнішній межі
        J_full = np.concatenate([[0.0], J, [J[-1]]])
        return (self.r_edges[1:]**2 * J_full[1:]
                - self.r_edges[:-1]**2 * J_full[:-1]) / (self.r**2 * self.dr)

    def _rhs(self, t, y):
        N = self.N
        u_b = np.maximum(y[:N], 1e-30)
        u_e = np.maximum(y[N:2*N], 1e-30)
        Psi = y[2*N:3*N]
        S = self.kappa_flow * self._f_Psi(Psi) * (u_b / self.rho_b_norm[0])
        div = self._divergence_flux(u_e)
        du_e = S - div
        du_b = -S
        dPsi = np.full(N, self.Theta0)
        return np.concatenate([du_b, du_e, dPsi])

    def _total_energy(self, y):
        N = self.N
        u_b = y[:N]
        u_e = y[N:2*N]
        return np.sum((u_b + u_e) * self.r**2 * self.dr)

    # ------------------------------------------------------------------
    # Головний цикл: квазі-стаціонарна еволюція + дискретні локальні події
    # ------------------------------------------------------------------
    def run(
        self,
        t_max_gyr: float = 13.8,
        event_cap: int = 3000,
        time_budget_seconds: float = 200.0,
        snapshots_target: int = 30,
        rtol: float = 1e-6,
        atol: float = 1e-18,
        max_step_s: float = 1e14,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Запускає симуляцію до t_max_gyr, або доки не буде досягнуто
        event_cap локальних подій, або доки не вичерпається бюджет часу
        виконання (реальні секунди, не модельний час).
        """
        N = self.N
        t_max = t_max_gyr * GYR

        y = np.concatenate([
            self.rho_b_norm,
            1e-6 * self.rho_b_norm,
            np.zeros(N),
        ])

        t_now = 0.0
        total_events = 0
        armed = np.ones(N, dtype=bool)
        event_radii = []
        snapshots = []
        snap_every = max(event_cap // snapshots_target, 1)

        t_start_wall = _time.time()

        def make_event(i):
            def ev(t, y):
                if not armed[i]:
                    return -1.0
                ub = max(y[i], 1e-30)
                ue = max(y[N + i], 1e-30)
                return ue / ub - self.beta_on
            ev.direction = 1
            ev.terminal = True
            return ev

        while (t_now < t_max and total_events < event_cap
               and _time.time() - t_start_wall < time_budget_seconds):

            events = [make_event(i) for i in range(N)]
            sol = solve_ivp(
                self._rhs, (t_now, t_max), y, method='BDF',
                rtol=rtol, atol=atol, events=events, max_step=max_step_s,
            )
            t_now = sol.t[-1]
            y = sol.y[:, -1]

            if sol.status != 1:
                if verbose:
                    print(f"[EtherGalaxyModel] Кінець інтервалу без нової "
                          f"події на t={t_now/GYR:.3f} Gyr.")
                break

            triggered = [i for i in range(N) if len(sol.t_events[i]) > 0]
            for i in triggered:
                total_events += 1
                Eb_i, Ee_i = y[i], y[N + i]
                tot = max(Eb_i + Ee_i, 1e-30)
                loss = self.Lc * tot
                Eb_i -= loss * (Eb_i / tot)
                Ee_i -= loss * (Ee_i / tot)
                y[i] = Eb_i + self.xi * Ee_i
                y[N + i] = (1 - self.xi) * Ee_i
                y[2*N + i] = (1 - self.kappa_reset) * y[2*N + i]
                armed[i] = False
                event_radii.append(self.r[i] / KPC)
                if total_events % snap_every == 0:
                    snapshots.append(y[N:2*N].copy())

            ratios_now = np.maximum(y[N:2*N], 1e-30) / np.maximum(y[:N], 1e-30)
            newly_armed = (~armed) & (ratios_now < self.beta_off_r)
            armed = armed | newly_armed

        energy_fraction = self._total_energy(y) / self._total_energy(
            np.concatenate([self.rho_b_norm, 1e-6 * self.rho_b_norm, np.zeros(N)])
        )
        radii_arr = np.array(event_radii)
        frac_outer = float(np.mean(radii_arr > 10)) if len(radii_arr) else float('nan')
        u_e_avg = np.mean(snapshots, axis=0) if snapshots else y[N:2*N]

        if verbose:
            print(f"[EtherGalaxyModel] t={t_now/GYR:.3f} Gyr "
                  f"({t_now/t_max*100:.1f}% від {t_max_gyr} Gyr), "
                  f"подій={total_events}, E_решта={energy_fraction*100:.2f}%, "
                  f"frac_outer={frac_outer*100:.1f}%, "
                  f"час обчислення={_time.time()-t_start_wall:.1f}с")

        return SimulationResult(
            t_reached_s=t_now,
            events_total=total_events,
            energy_fraction_remaining=energy_fraction,
            frac_outer=frac_outer,
            u_e_avg_snapshot=u_e_avg,
            final_state=y,
            event_radii_kpc=event_radii,
        )

    # ------------------------------------------------------------------
    # Переклад модельного профілю u_e(r) у нормовану криву DM(b)
    # ------------------------------------------------------------------
    def dm_shape(self, u_e_profile, b_values_kpc=(10, 30, 60, 100, 150, 200)):
        """
        Обчислює DM(b), нормоване на значення при найменшому b, шляхом
        інтегрування u_e(r) уздовж променя зору через сферичне гало,
        обмежене r200.

        УВАГА: це порівняння ФОРМИ (нормована крива), не абсолютних
        одиниць пк/см^3 — переклад u_e у фізичну густину електронів
        не реалізований (див. обмеження в докстрінгу модуля).
        """
        mask = self.r <= self.r200
        r_h = self.r[mask]
        u_e_h = np.maximum(u_e_profile[mask], 1e-30)

        def n_e(rr):
            return np.interp(rr, r_h, u_e_h, left=u_e_h[0], right=0.0)

        def DM_los(b):
            if b >= self.r200:
                return 0.0
            integrand = lambda s: n_e(np.sqrt(b**2 + s**2))
            val, _ = quad(integrand, 0, np.sqrt(max(self.r200**2 - b**2, 0)),
                          limit=100)
            return 2 * val

        dms = np.array([DM_los(b * KPC) for b in b_values_kpc])
        return dms / dms[0]


# ============================================================
# Приклад використання
# ============================================================
def _demo():
    print("=== Демонстрація: фінальна стабільна конфігурація ===\n")

    model = EtherGalaxyModel(N=128, v_wind=300e3)
    result = model.run(t_max_gyr=13.8, event_cap=1500, time_budget_seconds=90)

    print("\nФорма DM(b) (нормована, b=10..200 кпк):")
    dm = model.dm_shape(result.u_e_avg_snapshot)
    for b, d in zip((10, 30, 60, 100, 150, 200), dm):
        print(f"  b={b:4d} кпк:  DM/DM(10kpc) = {d:.3f}")

    print("\nПорівняння без вітру (базовий випадок):")
    model_base = EtherGalaxyModel(N=128, v_wind=0.0)
    result_base = model_base.run(t_max_gyr=13.8, event_cap=1500, time_budget_seconds=90)


if __name__ == "__main__":
    _demo()
