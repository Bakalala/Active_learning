import torch
from active_learning import plot_all, load_histories_from_ckpts, plot_all_histories, plot_all_rounds


# ckpt_path_mnist = "./active_learning/mnist.ckpt"

# ckpt = torch.load(ckpt_path_mnist, weights_only=False)
# history = ckpt["history"]
# plot_all(history)


pattern = "./active_learning/ISIC_*.ckpt"
all_hist = load_histories_from_ckpts(pattern)

# print(all_hist)

plot_all_histories(
    all_hist,
    metric="auc",
    save_path="plots/ISIC_test_auc_vs_labeled.png",   # relative path
)

plot_all_rounds(
    all_hist,
    metric="auc",
    save_path="plots/ISIC_test_auc_vs_round.png",     # relative path
)

plot_all_histories(
    all_hist,
    metric="acc",
    save_path="plots/ISIC_test_acc_vs_labeled.png",   # relative path
)

plot_all_rounds(
    all_hist,
    metric="acc",
    save_path="plots/ISIC_test_acc_vs_round.png",     # relative path
)

# # Plot test accuracy across strategies
# plot_all_histories(all_hist, metric="test_acc")
# plot_all_rounds(all_hist, metric="test_acc")

# # You can also compare other metrics:
# plot_all_histories(all_hist, metric="val_acc")
# plot_all_histories(all_hist, metric="train_loss", title="Train Loss vs # Labeled (all strategies)")