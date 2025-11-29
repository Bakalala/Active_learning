from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

from torchvision.transforms import functional as TF
import torchvision.transforms as T


from torchvision import datasets, transforms, models
from contextlib import nullcontext

from PIL import Image

from sklearn.metrics import roc_auc_score


import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os
import glob
import numpy as np

import pickle
from tqdm import tqdm

def save_ckpt(path, *, model, history, round_idx, labeled_idx, unlabeled_idx, val_idx, phase):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "round_idx": round_idx,
        "labeled_idx": labeled_idx,
        "unlabeled_idx": unlabeled_idx,
        "val_idx": val_idx,
        "phase": phase,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)
    print("Checkpoint saved at index ", round_idx)


def load_ckpt(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print("Loading from checkpoint")
    return ckpt

def plot_accuracy(history):
    rounds = history["round"]
    plt.figure(figsize=(6,4))
    plt.plot(rounds, history["train_acc"], marker='o', label="Train acc")
    plt.plot(rounds, history["val_acc"],   marker='o', label="Val acc")
    plt.plot(rounds, history["test_acc"],  marker='o', label="Test acc")
    plt.xlabel("Acquisition round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs acquisition round")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_loss(history):
    rounds = history["round"]
    plt.figure(figsize=(6,4))
    plt.plot(rounds, history["train_loss"], marker='o', label="Train loss")
    plt.xlabel("Acquisition round")
    plt.ylabel("Loss")
    plt.title("Train loss vs acquisition round")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_accuracy_vs_labels(history):
    labeled = history["labeled"]
    plt.figure(figsize=(6,4))
    plt.plot(labeled, history["val_acc"],  marker='o', label="Val acc")
    plt.plot(labeled, history["test_acc"], marker='o', label="Test acc")
    plt.xlabel("# Labeled examples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs labeled set size")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all(history):
    plot_accuracy(history)
    plot_loss(history)
    plot_accuracy_vs_labels(history)



def plot_all_histories(histories, metric="test_acc", title=None, save_path=None, y_lim = 0):
    """Plot a metric vs # labeled for multiple strategies."""
    plt.figure(figsize=(7, 5))

    if not histories:
        print("Warning: 'histories' is empty, nothing to plot.")

    for name, hist in histories.items():
        x = hist["labeled"]
        y = hist[metric]
        plt.plot(x, y, marker="o", label=name)

    plt.xlabel("# Labeled examples")
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(y_lim, 1.0)
    plt.title(title or f"{metric.replace('_',' ').title()} vs # Labeled (all strategies)")
    plt.grid(True)

    if histories:
        plt.legend()

    if save_path is not None:
        # IMPORTANT: use a relative path, not '/plots/...'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_all_histories] Saved plot to {save_path}")

    plt.close()  # free memory


def plot_all_rounds(histories, metric="test_acc", title=None, save_path=None, y_lim = 0):
    """Plot a metric vs acquisition round for multiple strategies."""
    plt.figure(figsize=(7, 5))

    if not histories:
        print("Warning: 'histories' is empty, nothing to plot.")

    for name, hist in histories.items():
        x = hist["round"]
        y = hist[metric]
        plt.plot(x, y, marker="o", label=name)

    plt.xlabel("Acquisition round")
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(y_lim, 1.0)
    plt.title(title or f"{metric.replace('_',' ').title()} vs Round (all strategies)")
    plt.grid(True)

    if histories:
        plt.legend()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_all_rounds] Saved plot to {save_path}")

    plt.close()


def load_histories_from_ckpts(pattern):
    """Load multiple histories from checkpoint files matching a glob pattern."""
    histories = {}
    for path in sorted(glob.glob(pattern)):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            hist = ckpt["history"]
            # Prefer the strategy name saved in history; else infer from filename
            name = hist.get("acq", os.path.splitext(os.path.basename(path))[0].split("mnist_")[-1])
            histories[name] = hist
        except Exception as e:
            print(f"[warn] Skipping {path}: {e}")
    return histories

# def plot_all_histories(histories, metric="test_acc", title=None):
#     """Plot a metric vs # labeled for multiple strategies."""
#     plt.figure(figsize=(7,5))
#     for name, hist in histories.items():
#         x = hist["labeled"]
#         y = hist[metric]
#         plt.plot(x, y, marker="o", label=name)
#     plt.xlabel("# Labeled examples")
#     plt.ylabel(metric.replace("_", " ").title())
#     plt.title(title or f"{metric.replace('_',' ').title()} vs # Labeled (all strategies)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

# def plot_all_rounds(histories, metric="test_acc", title=None):
#     """Plot a metric vs acquisition round for multiple strategies."""
#     plt.figure(figsize=(7,5))
#     for name, hist in histories.items():
#         x = hist["round"]
#         y = hist[metric]
#         plt.plot(x, y, marker="o", label=name)
#     plt.xlabel("Acquisition round")
#     plt.ylabel(metric.replace("_", " ").title())
#     plt.title(title or f"{metric.replace('_',' ').title()} vs Round (all strategies)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

def balanced_initial_split(dataset, n_labeled, n_val):
    labels = dataset.targets
    num_classes = len(torch.unique(labels))

    # collect and shuffle per class
    per_class = []
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(idx)
        per_class.append(idx)

    labeled_idx = []
    val_idx = []

    # compute how many to take per class
    lab_per = n_labeled // num_classes
    lab_rem = n_labeled % num_classes

    val_per = n_val // num_classes
    val_rem = n_val % num_classes

    # ----- ONE loop: for each class, take labeled first, then val -----
    for c in range(num_classes):
        cls = per_class[c]

        # labeled portion
        take_l = lab_per + (1 if c < lab_rem else 0)
        labeled_idx += cls[:take_l]

        # validation portion
        cls = cls[take_l:]
        take_v = val_per + (1 if c < val_rem else 0)
        val_idx += cls[:take_v]

        # update class list (remaining → unlabeled)
        per_class[c] = cls[take_v:]

    # unlabeled = everything left
    unlabeled_idx = []
    for c in range(num_classes):
        unlabeled_idx += per_class[c]

    return labeled_idx, val_idx, unlabeled_idx


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# MNIST nodel
class MNIST_CNN(nn.Module):
    # conv-relu -> conv-relu -> maxpool -> dropout -> dense-relu -> dropout -> dense
    def __init__(self, p1=0.25, p2=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p1)
        self.fc1   = nn.Linear(32 * 13 * 13, 128)
        self.drop2 = nn.Dropout(p2)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)  # use CrossEntropyLoss (includes softmax)

def train_one_round(model, loader, device, *, epochs, lr, weight_decay):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = total_steps = 0
    for ep in tqdm(range(epochs)):
        seen = correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            seen += yb.size(0)
            correct += (pred == yb).sum().item()
            total_loss += loss.item()
            total_steps += 1
        print(f"epoch {ep+1}/{epochs}  train_acc={correct/seen:.4f}")
    return (correct / seen ) , (total_loss / max(total_steps, 1))

@torch.inference_mode()  # faster + stricter than no_grad for pure inference
def evaluate(model, loader, device, use_amp=True):
    # Assume model is already on device outside; avoid .to(device) each call
    model.eval()
    is_cuda = (torch.device(device).type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_amp and is_cuda) else nullcontext()

    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(xb)

        pred = logits.argmax(dim=1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()

    return correct / max(total, 1)


def enable_mc_dropout(m: nn.Module):
    m.eval()
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            mod.train()
    return m

@torch.no_grad()
def mc_scores(model, dataset_subset, device, T=20, batch_size=256, mode="entropy"):

    loader = DataLoader(
        dataset_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, prefetch_factor=2, persistent_workers=True
    )

    enable_mc_dropout(model)
    model.to(device)

    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == 'cuda' else nullcontext()
    )

    eps = 1e-12
    out = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        B = xb.shape[0]

        # Vectorize MC passes: (B*T, ...)
        xb_rep = xb.repeat_interleave(T, dim=0)

        with autocast_ctx:
            logits = model(xb_rep)          # (B*T, C)
            p = logits.softmax(dim=1)       # (B*T, C)

        # (B, T, C) and mean over T
        p_btC = p.view(B, T, -1)
        p_mean = p_btC.mean(dim=1)          # (B, C)

        if mode == "entropy":
            # H[p]
            scores = -(p_mean * (p_mean + eps).log()).sum(dim=1)          # (B,)

        elif mode == "bald":
            # BALD = H[p] - mean_t H[p_t]
            H_mean = -(p_mean * (p_mean + eps).log()).sum(dim=1)          # (B,)
            H_each = -(p_btC * (p_btC + eps).log()).sum(dim=2)            # (B, T)
            scores = H_mean - H_each.mean(dim=1)                           # (B,)

        elif mode == "bald":
            # Variation ratio = 1 - (1/T) * max_c count_t [argmax_t == c]
            # Discrete predictions per MC pass: (B, T)
            y_bt = p_btC.argmax(dim=2)

            # One-hot vote counts per class: (B, C)
            counts = F.one_hot(y_bt, num_classes=p_btC.size(-1)).sum(dim=1)

            # Most common class count per sample: (B,)
            max_counts, _ = counts.max(dim=1)

            scores = 1.0 - max_counts.float() / T                         # (B,)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'entropy' or 'bald'.")

        out.append(scores.cpu())

    return torch.cat(out)


def acquire_once(model, train_set, unlabeled_idx, *, device, T, k, score_subset, test_batch, acquisition = 'random'):
    # sample a candidate subset for speed
    cand = random.sample(unlabeled_idx, k=min(score_subset, len(unlabeled_idx)))
    candidates = Subset(train_set, cand)

    if acquisition == 'random':
      picked = random.sample(cand, k=min(k, len(cand)))
    elif acquisition == 'entropy':
      scores = mc_scores(model, candidates, device, T=T, batch_size=test_batch, mode = 'entropy')
      top = torch.topk(scores, k=min(k, len(cand))).indices.tolist()
      picked = [cand[i] for i in top]
    elif acquisition == 'bald':
      scores = mc_scores(model, candidates, device, T=T, batch_size=test_batch, mode = 'bald')
      top = torch.topk(scores, k=min(k, len(cand))).indices.tolist()
      picked = [cand[i] for i in top]
    elif acquisition == 'var':
      scores = mc_scores(model, candidates, device, T=T, batch_size=test_batch, mode = 'bald')
      top = torch.topk(scores, k=min(k, len(cand))).indices.tolist()
      picked = [cand[i] for i in top]

    # update pools
    unlab_set = set(unlabeled_idx)
    for i in picked: unlab_set.discard(i)
    return picked, list(unlab_set)

def pick_batch_size(n):
    if n < 64:        return 16
    if n < 256:       return 32
    if n < 1024:      return 64
    if n < 4096:      return 128
    else:             return 256

def active_learning_loop(model, train_set, test_set,
                         labeled_idx, val_idx, unlabeled_idx, CFG,resume = True, ckpt_path = ''):

    history = {
        "round": [],
        "labeled": [],
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "test_acc": [],
    }

    start_round = 0
    resume_phase = 'post_acq'
    last_ckpt_round = -1
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ckpt = load_ckpt(ckpt_path, model)
        history = ckpt["history"]
        last_ckpt_round = ckpt["round_idx"]
        start_round = last_ckpt_round if resume_phase == "post_eval" else last_ckpt_round + 1
        labeled_idx   = ckpt["labeled_idx"]
        unlabeled_idx = ckpt["unlabeled_idx"]
        val_idx       = ckpt["val_idx"]
        resume_phase   = ckpt["phase"]
        # restore RNG
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.random.set_rng_state(ckpt["rng_torch"])
        if ckpt.get("rng_torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])
        print(f"[resume] starting from round {start_round}")

        if CFG.acquisition_rounds > last_ckpt_round + 1:
            print("[resume] Checkpoint marked 'done' but config requests MORE rounds.")
            print("[resume] Will continue training from last checkpoint round.")
            resume_phase = "post_acq"
            start_round = last_ckpt_round + 1
        else:
            print("[resume] Run already finished under this configuration.")
            return history
        print('resume_phase is ', resume_phase)


    val_loader = DataLoader(Subset(train_set, val_idx),
                            batch_size=CFG.test_batch_size, shuffle=False,
                            num_workers=2, prefetch_factor=2)
    test_loader = DataLoader(test_set,batch_size=CFG.test_batch_size, shuffle=False,
                             num_workers=2, prefetch_factor=2)

    for r in range(start_round,CFG.acquisition_rounds):
        print(f"\n=== Round {r+1}/{CFG.acquisition_rounds} ===")

        print("DBG:", "r", r, "last_ckpt_round", last_ckpt_round, "resume_phase", resume_phase)
        if resume_phase == "post_eval" and r == last_ckpt_round:
            print("[resume] skipping training; going straight to acquisition.")
        else:
            # dynamic batch size
            bs = pick_batch_size(len(labeled_idx))
            # loaders for current labeled set
            labeled_loader = DataLoader(Subset(train_set, labeled_idx),
                                        batch_size=bs, shuffle=True)

            # train + eval
            train_acc, train_loss = train_one_round(model, labeled_loader, CFG.device,
                            epochs=CFG.epochs_per_round, lr=CFG.lr, weight_decay=CFG.weight_decay)
            val_acc  = evaluate(model, val_loader,  CFG.device)
            test_acc = evaluate(model, test_loader, CFG.device)
            print(f"val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

            history["round"].append(r)
            history["labeled"].append(len(labeled_idx))
            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)

            if ckpt_path:
                save_ckpt(ckpt_path, model=model, history=history, round_idx=r,
                          labeled_idx=labeled_idx, unlabeled_idx=unlabeled_idx, val_idx=val_idx,
                          phase="post_eval")

        # don't acquire on the last round since won't be used
        if r == CFG.acquisition_rounds - 1 or not unlabeled_idx:
            if ckpt_path:
              save_ckpt(ckpt_path, model=model, history=history, round_idx=r,
                      labeled_idx=labeled_idx, unlabeled_idx=unlabeled_idx, val_idx=val_idx,
                      phase="done")
            break

        # acquire CFG.acquisition_size new labels
        picked, unlabeled_idx = acquire_once(
            model, train_set, unlabeled_idx,
            device=CFG.device, T=CFG.mc_passes,
            k=CFG.acquisition_size, score_subset=CFG.score_subset,
            test_batch=CFG.test_batch_size, acquisition = CFG.acq
        )
        labeled_idx += picked
        print(f"acquired {len(picked)} → labeled={len(labeled_idx)}  unlabeled={len(unlabeled_idx)}")


        # checkpoint after every round
        if ckpt_path:
            save_ckpt(ckpt_path, model=model, history=history, round_idx=r,
                      labeled_idx=labeled_idx, unlabeled_idx=unlabeled_idx, val_idx=val_idx, phase ="post_acq")

    return history
