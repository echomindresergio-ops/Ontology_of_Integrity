"""
Temporal clustering (history-based) -- архітектурна зміна, не заміна статистики
====================================================================================
Ідея: кожен агент веде ковзний буфер останніх history_window сирих top-1
позицій. Щоцикл буфер кластеризується (gap-based, як у Прототипі A2),
кластери з масою >= min_cluster_mass лишаються. Порівняння МІЖ агентами
відбувається на РІВНІ МНОЖИН кластерів (чи є пара центрів (A,B) в межах
eps), а не на рівні одноциклового пікового вибору. Це має дозволити
виявляти узгодженість на різних піках, навіть якщо в ПОТОЧНОМУ циклі
агенти дивляться на різні (з 3) піки -- бо історія за 25-50 циклів
повинна містити відвідування всіх пікыв порівнянної сили.

ВІДОМИЙ РИЗИК (перевіряється пілотом): буфер ковзає по 1 точці за цикл,
тому кластерна структура майже не змінюється цикл-до-циклу -- якщо
збіг знайдено раз, він, ймовірно, повторюватиметься ~history_window
циклів поспіль. Це може РІЗКО пришвidshvidshvidшити накопичення
підтверджень (добре для TPR), але так само різко підняти FPR під шумом
(кластери шумових точок так само стабільні протягом history_window).
"""
import sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/outputs")
import echomindre_v5_v3_grid_sweep_top1 as g
import echomindre_v5_prototype as m


def cluster_buffer(buffer, cluster_radius, min_cluster_mass):
    positions = np.sort(np.array(buffer))
    n = len(positions)
    clusters = []
    cur = [positions[0]]
    for i in range(1, n):
        if positions[i] - positions[i - 1] > cluster_radius:
            clusters.append(np.array(cur))
            cur = []
        cur.append(positions[i])
    clusters.append(np.array(cur))
    result = []
    for c in clusters:
        mass = len(c) / n
        if mass >= min_cluster_mass:
            result.append((float(np.mean(c)), mass))
    return result


def run_pipeline_temporal_cluster(p, clean_sig, noise_level, temp, n_runs, seed0, eps,
                                   history_window, cluster_radius, min_cluster_mass):
    n = g.n
    n_confirmed = 0
    accepted = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed0 + i)
        heldout_n = int(p["heldout_frac"] * n)
        train_mask = np.ones(n, dtype=bool)
        idx = np.arange(n)
        rng.shuffle(idx)
        train_mask[idx[:heldout_n]] = False
        memory = m.MemorySlots(p["n_templates"], sigma=p["mem_kernel_sigma"])
        buf_a, buf_b = [], []
        n_acc = 0
        for cycle in range(p["cycles_per_run"]):
            a = g.raw_candidate_top1(clean_sig, noise_level, temp, rng, train_mask)
            b = g.raw_candidate_top1(clean_sig, noise_level, temp, rng, train_mask)
            buf_a.append(a); buf_b.append(b)
            if len(buf_a) > history_window:
                buf_a.pop(0); buf_b.pop(0)
            if len(buf_a) >= max(5, history_window // 2):
                clusters_a = cluster_buffer(buf_a, cluster_radius, min_cluster_mass)
                clusters_b = cluster_buffer(buf_b, cluster_radius, min_cluster_mass)
                matched = False
                for ca, _ in clusters_a:
                    for cb, _ in clusters_b:
                        if abs(ca - cb) <= eps:
                            matched = True
                            cand = (ca + cb) / 2
                            break
                    if matched:
                        break
                if matched:
                    n_acc += 1
                    memory.add_or_reinforce(cand, 0.5, p["sim_threshold"], p["memory_forget_lambda"])
            memory.confirm_validated(p["validation_k"])
        accepted.append(n_acc)
        if memory.n_confirmed() > 0:
            n_confirmed += 1
    return n_confirmed / n_runs, float(np.mean(accepted))


if __name__ == "__main__":
    p = dict(g.BASE_P)
    p["validation_k"] = 4
    p["sim_threshold"] = 0.28
    p["n_templates"] = 3
    eps = 0.0120
    history_window = 25
    cluster_radius = 0.05
    min_cluster_mass = 0.10
    temps = [0.2, 0.5, 1.0]
    noises = [0.1, 0.5, 1.0]

    seed_t = 17
    c, a, w = m.true_templates(g.grid, p["n_templates"], seed=seed_t)
    clean_sig = m.clean_signal(g.grid, c, a, w)
    clean_null = np.zeros_like(g.grid)

    print(f"=== PILOT: seed=17, history_window={history_window}, cluster_radius={cluster_radius}, "
          f"min_cluster_mass={min_cluster_mass}, eps={eps} ===")
    worst_fpr = 0.0
    for t in temps:
        fpr, acc = run_pipeline_temporal_cluster(p, clean_null, 1.0, t, 80, 500000 + int(t * 100), eps,
                                                    history_window, cluster_radius, min_cluster_mass)
        worst_fpr = max(worst_fpr, fpr)
        print(f"  [null] temp={t}: FPR={fpr:.4f} (accepted_mean={acc:.2f})")
    print(f"  worst_FPR = {worst_fpr:.4f}")

    worst_tpr = 1.0
    for t in temps:
        for noise in noises:
            tpr, acc = run_pipeline_temporal_cluster(p, clean_sig, noise, t, 60, 510000 + int(t * 100) + int(noise * 1000),
                                                        eps, history_window, cluster_radius, min_cluster_mass)
            worst_tpr = min(worst_tpr, tpr)
            print(f"  [signal] temp={t} noise={noise}: TPR={tpr:.4f} (accepted_mean={acc:.2f})")
    print(f"  worst_TPR = {worst_tpr:.4f}")
