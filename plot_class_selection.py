from active_learning import load_histories_from_ckpts , plot_all_histories, plot_all_rounds
import numpy as np
import matplotlib.pyplot as plt

pattern = "./active_learning/mnist_bin_*.ckpt"
all_hist = load_histories_from_ckpts(pattern)


for name, history in all_hist.items():
    print('plotting history for', name)
    acq_counts = np.array(history["acquired_counts"])
    rounds = np.arange(1, acq_counts.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(acq_counts.shape[0])
    for c in range(acq_counts.shape[1]):
        plt.bar(rounds, acq_counts[:, c], bottom=bottom, label=f"Class {c}")
        bottom += acq_counts[:, c]

    plt.xlabel("Acquisition round")
    plt.ylabel("# samples acquired")
    plt.title(f"MNIST: class distribution per round ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/mnist_bin_{name}_class_distribution.png")

for name, history in all_hist.items():
    print('plotting history for', name)
    acq_counts = np.array(history["acquired_counts"])   # shape [rounds, classes]

    # Make cumulative
    acq_cum = acq_counts.cumsum(axis=0)

    rounds = np.arange(1, acq_cum.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(acq_cum.shape[0])

    for c in range(acq_cum.shape[1]):
        plt.bar(rounds, acq_cum[:, c], bottom=bottom, label=f"Class {c}")
        bottom += acq_cum[:, c]

    plt.xlabel("Acquisition round")
    plt.ylabel("Cumulative # samples acquired")
    plt.title(f"MNIST: cumulative acquired samples per class ({name})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"plots/mnist_{name}_class_distribution_cumulative.png")
    plt.close()

ylimit = 0.8

plot_all_histories(
    all_hist,
    metric="test_acc",
    save_path="plots/MNIST_bin_test_acc_vs_labeled.png",   # relative path
    y_lim = ylimit
)

plot_all_rounds(
    all_hist,
    metric="test_acc",
    save_path="plots/MNIST_bin_test_acc_vs_round.png",     # relative path
    y_lim = ylimit

)