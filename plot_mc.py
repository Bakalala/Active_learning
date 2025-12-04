from active_learning import load_histories_from_ckpts , plot_all_histories, plot_all_rounds
import numpy as np
import matplotlib.pyplot as plt

pattern = "./active_learning/mnist_mc_*.ckpt"
all_hist = load_histories_from_ckpts(pattern)


ylimit = 0.8

plot_all_histories(
    all_hist,
    metric="test_acc",
    save_path="plots/MNIST_mc_test_acc_vs_labeled.png",   # relative path
    y_lim = ylimit
)

plot_all_rounds(
    all_hist,
    metric="test_acc",
    save_path="plots/MNIST_mc_test_acc_vs_round.png",     # relative path
    y_lim = ylimit

)