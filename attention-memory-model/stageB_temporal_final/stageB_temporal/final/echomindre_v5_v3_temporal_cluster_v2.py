"""
Temporal clustering v2 -- виправлення FPR-патології з пілота v1
======================================================================
Зміни відносно v1 (за інструкцією):
  1. Порівнюємо ЛИШЕ домінантний кластер кожного агента (найбільша
     кількість точок), а не всі пари кластерів.
  2. min_cluster_points -- абсолютний поріг (кількість точок), а не
     частка від поточного розміру буфера.
  3. Cooldown: після підтвердження кластера в певній позиції, заборона
     повторного підтвердження в тому ж місці протягом cooldown_window
     циклів (щоб персистентний через ковзання буфера кластер не
     тригерив підтвердження щоцикл).
"""
import sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/outputs")
import echomindre_v5_v3_grid_sweep_top1 as g
import echomindre_v5_prototype as m


def cluster_buffer_abs(buffer, cluster_radius, min_cluster_points):
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
    valid = [c for c in clusters if len(c) >= min_cluster_points]
    if not valid:
        return None, 0, 0
    dom = max(valid, key=lambda c: len(c))
    return float(np.mean(dom)), len(dom), len(dom) / n


def run_pipeline_temporal_v2(p, clean_sig, noise_level, temp, n_runs, seed0, eps,
                              history_window, cluster_radius, min_cluster_points, cooldown_window,
                              collect_diag=False):
    n = g.n
    n_confirmed = 0
    accepted = []
    diag_rows = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed0 + i)
        heldout_n = int(p["heldout_frac"] * n)
        train_mask = np.ones(n, dtype=bool)
        idx = np.arange(n)
        rng.shuffle(idx)
        train_mask[idx[:heldout_n]] = False
        memory = m.MemorySlots(p["n_templates"], sigma=p["mem_kernel_sigma"])
        buf_a, buf_b = [], []
        cooldowns = []  # list of (center, expire_cycle)
        n_acc = 0
        for cycle in range(p["cycles_per_run"]):
            a = g.raw_candidate_top1(clean_sig, noise_level, temp, rng, train_mask)
            b = g.raw_candidate_top1(clean_sig, noise_level, temp, rng, train_mask)
            buf_a.append(a); buf_b.append(b)
            if len(buf_a) > history_window:
                buf_a.pop(0); buf_b.pop(0)

            chosen = None
            if len(buf_a) >= min_cluster_points:
                dom_a, npts_a, mass_a = cluster_buffer_abs(buf_a, cluster_radius, min_cluster_points)
                dom_b, npts_b, mass_b = cluster_buffer_abs(buf_b, cluster_radius, min_cluster_points)
                if dom_a is not None and dom_b is not None and abs(dom_a - dom_b) <= eps:
                    chosen = (dom_a + dom_b) / 2
                if collect_diag and i == 0:
                    diag_rows.append(dict(cycle=cycle, dom_a=dom_a, npts_a=npts_a, mass_a=mass_a,
                                           dom_b=dom_b, npts_b=npts_b, mass_b=mass_b, chosen=chosen))

            if chosen is not None:
                # cooldown check
                on_cooldown = any(abs(chosen - c) <= cluster_radius and cycle < exp for c, exp in cooldowns)
                if not on_cooldown:
                    n_acc += 1
                    memory.add_or_reinforce(chosen, 0.5, p["sim_threshold"], p["memory_forget_lambda"])
                    cooldowns.append((chosen, cycle + cooldown_window))
            # prune expired cooldowns
            cooldowns = [(c, exp) for c, exp in cooldowns if exp > cycle]
            memory.confirm_validated(p["validation_k"])
        accepted.append(n_acc)
        if memory.n_confirmed() > 0:
            n_confirmed += 1
    result = (n_confirmed / n_runs, float(np.mean(accepted)))
    if collect_diag:
        return result, diag_rows
    return result


def calibrate_eps_temporal(p, temp, n_null, history_window, cluster_radius, min_cluster_points, cooldown_window,
                            target=0.05, steps=9, lo=1e-6, hi=0.02, seed_base=700000):
    for step in range(steps):
        mid = (lo + hi) / 2
        clean_null = np.zeros_like(g.grid)
        fpr, acc = run_pipeline_temporal_v2(p, clean_null, 1.0, temp, n_null, seed_base + step, mid,
                                             history_window, cluster_radius, min_cluster_points, cooldown_window)
        if fpr > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    p = dict(g.BASE_P)
    p["validation_k"] = 4
    p["sim_threshold"] = 0.28
    p["n_templates"] = 3
    history_window = 25
    cluster_radius = 0.05
    min_cluster_points = 3
    cooldown_window = 12
    temps = [0.2, 0.5, 1.0]
    noises = [0.1, 0.5, 1.0]

    seed_t = 17
    c, a, w = m.true_templates(g.grid, p["n_templates"], seed=seed_t)
    clean_sig = m.clean_signal(g.grid, c, a, w)
    clean_null = np.zeros_like(g.grid)

    print(f"=== PILOT v2: seed=17, history_window={history_window}, cluster_radius={cluster_radius}, "
          f"min_cluster_points={min_cluster_points}, cooldown_window={cooldown_window} ===")

    # binary search eps at temp=0.5, then verify worst-case
    eps = calibrate_eps_temporal(p, 0.5, 150, history_window, cluster_radius, min_cluster_points, cooldown_window)
    print(f"calibrated eps (temp=0.5, N=150) = {eps:.6f}")

    worst_fpr = 0.0
    for t in temps:
        fpr, acc = run_pipeline_temporal_v2(p, clean_null, 1.0, t, 150, 800000 + int(t * 100), eps,
                                             history_window, cluster_radius, min_cluster_points, cooldown_window)
        worst_fpr = max(worst_fpr, fpr)
        print(f"  [null] temp={t}: FPR={fpr:.4f} (accepted_mean={acc:.2f})")
    print(f"  worst_FPR = {worst_fpr:.4f}")

    worst_tpr = 1.0
    for t in temps:
        for noise in noises:
            tpr, acc = run_pipeline_temporal_v2(p, clean_sig, noise, t, 100, 810000 + int(t * 100) + int(noise * 1000),
                                                 eps, history_window, cluster_radius, min_cluster_points, cooldown_window)
            worst_tpr = min(worst_tpr, tpr)
            print(f"  [signal] temp={t} noise={noise}: TPR={tpr:.4f} (accepted_mean={acc:.2f})")
    print(f"  worst_TPR = {worst_tpr:.4f}")
