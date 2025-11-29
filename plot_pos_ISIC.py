import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle


def plot_isic_histories(all_histories, initial_pos=20, title_prefix="ISIC 2016"):
    """
    all_histories: dict[str, history or list[history]]
        Each history dict must have keys:
          - "auc": list of AUC per round (includes round 0, before any acquisitions)
          - "pos_acquired": list of positives acquired at each acquisition step
            (usually length = len(auc) - 1)

        If value is a single dict, it's treated as one run.
        If value is a list of dicts, they are averaged.

    initial_pos: int
        Number of positive examples already in the initial labeled set (step 0).
    """

    def ensure_runs(entry):
        # Allow all_histories["bald"] to be either a dict or a list of dicts
        if isinstance(entry, dict):
            return [entry]
        elif isinstance(entry, list):
            return entry
        else:
            raise TypeError(
                "Each value in all_histories must be a history dict or list of dicts."
            )

    def curves_from_history(hist):
        """
        Returns (auc, pos_cum) for a single run.
        auc: shape (T,)
        pos_cum: shape (T,) with pos_cum[0] = initial_pos
        """
        auc = np.asarray(hist["auc"], dtype=float)
        pos_acq = np.asarray(hist["pos_acquired"], dtype=float)

        n_steps = len(auc)
        pos_cum = np.empty(n_steps, dtype=float)
        pos_cum[0] = initial_pos
        if n_steps > 1:
            # pos_acq usually has length n_steps-1 (no acquisition after final eval)
            pos_cum[1:] = initial_pos + np.cumsum(pos_acq[: n_steps - 1])
        return auc, pos_cum

    def stack_mean_se(runs, which="auc"):
        """
        runs: list[history]
        which: "auc" or "pos"
        Returns (steps, mean, se).
        """
        auc_list = []
        pos_list = []
        for h in runs:
            auc, pos_cum = curves_from_history(h)
            auc_list.append(auc)
            pos_list.append(pos_cum)

        # Make sure all runs have the same length – truncate to shortest.
        min_len = min(len(a) for a in auc_list)
        auc_arr = np.stack([a[:min_len] for a in auc_list], axis=0)
        pos_arr = np.stack([p[:min_len] for p in pos_list], axis=0)

        arr = auc_arr if which == "auc" else pos_arr

        mean = arr.mean(axis=0)
        if arr.shape[0] > 1:
            se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        else:
            se = np.zeros_like(mean)
        steps = np.arange(len(mean))  # acquisition steps 0..T-1
        return steps, mean, se

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_auc, ax_pos = axes

    # Consistent ordering + pretty display names
    raw_methods = list(all_histories.keys())
    preferred_order = ["random", "bald"]
    methods = [m for m in preferred_order if m in raw_methods] + [
        m for m in raw_methods if m not in preferred_order
    ]
    display_name = {
        "random": "uniform",
        "bald": "BALD",
    }

    for method in methods:
        runs = ensure_runs(all_histories[method])

        # AUC plot
        steps_auc, mean_auc, se_auc = stack_mean_se(runs, which="auc")
        ax_auc.plot(steps_auc, mean_auc, label=display_name.get(method, method))
        ax_auc.fill_between(
            steps_auc, mean_auc - se_auc, mean_auc + se_auc, alpha=0.2
        )

        # # positives plot
        steps_pos, mean_pos, se_pos = stack_mean_se(runs, which="pos")
        ax_pos.plot(steps_pos, mean_pos, label=display_name.get(method, method))
        ax_pos.fill_between(
            steps_pos, mean_pos - se_pos, mean_pos + se_pos, alpha=0.2
        )

    # Styling like the paper
    ax_auc.set_xlabel("Acquisition steps")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_title(f"{title_prefix}: AUC vs acquisition step")
    ax_auc.legend()

    ax_pos.set_xlabel("Acquisition steps")
    ax_pos.set_ylabel("# positive examples acquired")
    ax_pos.set_title(f"{title_prefix}: positives vs acquisition step")
    ax_pos.legend()

    fig.tight_layout()
    return fig, axes


# ==== load and plot ====

pkl_path = "active_learning/ISIC_all_histories.pkl"
with open(pkl_path, "rb") as f:
    all_histories = pickle.load(f)

fig, axes = plot_isic_histories(
    all_histories, initial_pos=20, title_prefix="ISIC 2016"
)

save_path = "plots/ISIC_pos_count_og.png"
# IMPORTANT: use a relative path, not '/plots/...'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches="tight", dpi=200)
print(f"[plot_all_histories] Saved plot to {save_path}")
