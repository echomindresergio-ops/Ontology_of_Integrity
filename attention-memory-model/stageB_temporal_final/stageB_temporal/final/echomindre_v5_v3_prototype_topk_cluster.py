"""
Прототип A: top-k + кластеризація (заміна чистого top-1)
==============================================================
Мета: виправити фрагментацію доказів між кількома рознесеними піками
порівнянної амплітуди (знайдено на seed_templates=17,19 -- worst_TPR
впав до 0.50-0.58 при noise=1.0, для top-1, незалежно від sim_threshold).

Логіка (Прототип A, як описано в плані):
  1. Беремо top-k=3 позицій за вагою уваги за цикл (для кожного агента).
  2. Якщо розкид (max-min) цих k позицій <= cluster_radius -> вважаємо
     їх одним кластером, кандидат = зважене середнє (вагами уваги).
  3. Якщо розкид > cluster_radius -> позиції належать різним пікам,
     НЕ підсилюємо жоден слот цього циклу (candidate=None).
  4. Порівняння між агентами: обидва мають дати НЕ-None кандидата,
     і |cand_A - cand_B| <= eps -> приймається.
"""
import sys
import numpy as np
sys.path.insert(0, "/mnt/user-data/outputs")
import echomindre_v5_v3_grid_sweep_top1 as g
import echomindre_v5_prototype as m


def raw_candidate_dominant_cluster(clean, noise_level, temp, rng, train_mask, top_frac, gap_threshold):
    """Бере top_frac (напр. 20%) точок за вагою, розбиває їх на контактні
    кластери за розривом позицій (gap_threshold), обирає кластер з
    найбільшою СУМАРНОЮ вагою (не просто найближчий до argmax), кандидат =
    зважене середнє точок цього домінантного кластера. Це замінює і чистий
    top-1 (немає стрибків між одиночними шумовими точками), і старий
    top-20%-average (немає усереднення між РІЗНИМИ, незв'язаними піками)."""
    y = m.observe(clean, noise_level, rng)
    pi = m.attention_distribution(np.where(train_mask, y, y.min()), temp)
    train_positions = np.where(train_mask)[0]
    k = max(1, int(top_frac * len(train_positions)))
    top_idx = train_positions[np.argsort(-pi[train_positions])[:k]]
    positions = g.grid[top_idx]
    weights = pi[top_idx]
    order = np.argsort(positions)
    positions = positions[order]
    weights = weights[order]
    # розбити на кластери за розривом
    clusters = []
    cur_pos = [positions[0]]
    cur_w = [weights[0]]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] > gap_threshold:
            clusters.append((np.array(cur_pos), np.array(cur_w)))
            cur_pos, cur_w = [], []
        cur_pos.append(positions[i])
        cur_w.append(weights[i])
    clusters.append((np.array(cur_pos), np.array(cur_w)))
    # обрати кластер з найбільшою сумарною вагою
    best = max(clusters, key=lambda c: c[1].sum())
    cand = float(np.sum(best[0] * best[1]) / np.sum(best[1]))
    n_clusters = len(clusters)
    return cand, n_clusters


def run_pipeline_dominant_cluster(p, clean_sig, noise_level, temp, n_runs, seed0, eps, top_frac, gap_threshold):
    n = g.n
    n_confirmed = 0
    accepted = []
    n_clusters_list = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed0 + i)
        heldout_n = int(p["heldout_frac"] * n)
        train_mask = np.ones(n, dtype=bool)
        idx = np.arange(n)
        rng.shuffle(idx)
        train_mask[idx[:heldout_n]] = False
        memory = m.MemorySlots(p["n_templates"], sigma=p["mem_kernel_sigma"])
        n_acc = 0
        for _ in range(p["cycles_per_run"]):
            a, nc_a = raw_candidate_dominant_cluster(clean_sig, noise_level, temp, rng, train_mask, top_frac, gap_threshold)
            b, nc_b = raw_candidate_dominant_cluster(clean_sig, noise_level, temp, rng, train_mask, top_frac, gap_threshold)
            n_clusters_list.append((nc_a + nc_b) / 2)
            if abs(a - b) <= eps:
                n_acc += 1
                memory.add_or_reinforce(a, 0.5, p["sim_threshold"], p["memory_forget_lambda"])
            memory.confirm_validated(p["validation_k"])
        accepted.append(n_acc)
        if memory.n_confirmed() > 0:
            n_confirmed += 1
    return (n_confirmed / n_runs, float(np.mean(accepted)), float(np.mean(n_clusters_list)))


if __name__ == "__main__":
    p = dict(g.BASE_P)
    p["validation_k"] = 4
    p["sim_threshold"] = 0.28
    p["n_templates"] = 3
    eps = 0.0120
    top_frac = 0.20
    temps = [0.2, 0.5, 1.0]
    noises = [0.1, 0.5, 1.0]

    import pandas as pd
    rows = []
    for seed_t in [17]:
        c, a, w = m.true_templates(g.grid, p["n_templates"], seed=seed_t)
        clean_sig = m.clean_signal(g.grid, c, a, w)
        clean_null = np.zeros_like(g.grid)
        for gap_threshold in [0.02, 0.05, 0.08]:
            worst_fpr = 0.0
            for t in temps:
                fpr, acc_n, nclust_n = run_pipeline_dominant_cluster(
                    p, clean_null, 1.0, t, 250, 400000 + int(gap_threshold * 1000) + int(t * 100), eps, top_frac, gap_threshold)
                worst_fpr = max(worst_fpr, fpr)
                rows.append(dict(seed_templates=seed_t, gap_threshold=gap_threshold, temp=t, noise=None,
                                  phase="null_scenario", metric="FPR", value=fpr, accepted_mean=acc_n,
                                  n_clusters_mean=nclust_n))
            worst_tpr = 1.0
            for t in temps:
                for noise in noises:
                    tpr, acc_s, nclust_s = run_pipeline_dominant_cluster(
                        p, clean_sig, noise, t, 150, 410000 + int(gap_threshold * 1000) + int(t * 100) + int(noise * 1000), eps, top_frac, gap_threshold)
                    worst_tpr = min(worst_tpr, tpr)
                    rows.append(dict(seed_templates=seed_t, gap_threshold=gap_threshold, temp=t, noise=noise,
                                      phase="signal_scenario", metric="TPR", value=tpr, accepted_mean=acc_s,
                                      n_clusters_mean=nclust_s))
            print(f"seed={seed_t} gap_threshold={gap_threshold}: worst_FPR={worst_fpr:.4f} worst_TPR={worst_tpr:.4f}")

    pd.DataFrame(rows).to_csv("/mnt/user-data/outputs/v3_prototypeA2_dominant_cluster_seed17.csv", index=False)
