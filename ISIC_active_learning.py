from active_learning import *
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ISIC_BASE = "./data/ISIC2016"

class ISICSimpleDataset(Dataset):

    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.items = []     # list of (img_path, label)
        self.labels = []    # parallel list of int labels (0/1)

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                image_id, label_str = parts[0], parts[1]
                label_str = 0 if label_str == 'benign' else 1
                label = int(label_str)
                jpg = os.path.join(img_dir, image_id + ".jpg")
                img_path = jpg
                self.items.append((img_path, label))
                self.labels.append(label)
        if len(self.items) == 0:
            raise RuntimeError(f"No items found from {csv_path} under {img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class PosFlipAug(Dataset):
    def __init__(self, base_subset):
        self.base = base_subset
        # cache labels from base subset
        self.pos_idx = [i for i in range(len(base_subset)) if base_subset[i][1] == 1]
        self.views = [("orig", None, None, i) for i in range(len(base_subset))]
        self.views += [("h", True,  False, i) for i in self.pos_idx]
        self.views += [("v", False, True,  i) for i in self.pos_idx]

    def __len__(self): return len(self.views)

    def __getitem__(self, idx):
        kind, do_h, do_v, base_i = self.views[idx]
        x,y = self.base[base_i]
        if do_h: x = TF.hflip(x)
        if do_v: x = TF.vflip(x)
        return x,y


def build_vgg16_binary(num_classes=2, p=0.5):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # replace classifier head
    in_features = m.classifier[0].in_features  # 25088 for VGG16
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.Linear(4096, num_classes),
    )
    return m

# reset the model to it's original parameters after each acquisition step
def reset_model(model, initial_state):
    model.load_state_dict(initial_state, strict=True)

def make_balanced_test_and_initial(train_full, seed=42):
    y = np.array(train_full.labels, dtype=int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(pos); rng.shuffle(neg)

    # test set
    test_pos = pos[:100];  test_neg = neg[:100]
    # remaining pool
    pool_pos = pos[100:];  pool_neg = neg[100:]

    # initial labeled (80/20)
    init_pos = pool_pos[:20];  init_neg = pool_neg[:80]
    pool_pos = pool_pos[20:];  pool_neg = pool_neg[80:]

    test_idx      = np.concatenate([test_pos, test_neg]).tolist()
    labeled_idx   = np.concatenate([init_pos, init_neg]).tolist()
    unlabeled_idx = np.concatenate([pool_pos, pool_neg]).tolist()
    return labeled_idx, unlabeled_idx, test_idx



def evaluate_auc_mc(model, loader, device, T=20):
    model.to(device).eval()
    enable_mc_dropout(model)   # keep dropout active

    all_probs = []
    all_labels = []

    with torch.no_grad():  # <<< crucial for memory
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # Collect T stochastic forward passes
            preds_T = []
            for _ in range(T):
                logits = model(xb)
                probs = logits.softmax(dim=1)[:, 1]  # prob of malignant class
                preds_T.append(probs)

            # Shape: (T, B)
            preds_T = torch.stack(preds_T, dim=0)

            # Mean probability over MC samples
            mean_probs = preds_T.mean(dim=0)  # (B,)

            all_probs.append(mean_probs.cpu())
            all_labels.append(yb.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(all_labels, all_probs)
    return auc


# ISIC config
class Config:
    def __init__(self, acq = 'random', acquisition_rounds = 5):
        self.acq = acq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training settings
        self.batch_size = 8
        self.test_batch_size = 64
        self.epochs_per_round = 50
        self.lr = 1e-4
        self.weight_decay = 5e-4

        # AL

        self.initial_labeled = 20
        self.validation_set_size = 100
        self.acquisition_size = 100
        self.acquisition_rounds = acquisition_rounds
        self.mc_passes = 20
        self.score_subset = 5000
        self.seed = 42 # Used 42 and 10
        


        # ckpt file unique to strategy
        self.ckpt_path = f"./active_learning/ISIC_{acq}.ckpt"

def clean_cuda():
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()



def run_active_learning_ISIC(train_full, labeled_idx, unlabeled_idx, test_idx, CFG):

    # reset to pretrained at the start of each round (as in the paper)
    model = build_vgg16_binary()
    initial_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    history = {
        "round": [],
        "labeled": [],
        "auc": [],
        "acc": [],
        "pos_acquired": [],
        "acq": CFG.acq,  # track which acquisition this history belongs to
    }

    test_loader = DataLoader(
        Subset(train_full, test_idx),
        batch_size=CFG.test_batch_size,
        shuffle=True,
        num_workers=2
    )

    for r in range(CFG.acquisition_rounds):
        print(f"\n=== Round {r+1}/{CFG.acquisition_rounds} (acq={CFG.acq}) ===")
        reset_model(model, initial_state)

        # build augmented training data for current labeled set
        base_subset = Subset(train_full, labeled_idx)
        aug_dataset = PosFlipAug(base_subset)
        train_loader = DataLoader(
            aug_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=2
        )

        # weight decay: (1 - p) * l^2 / N with p=0.5, l^2=0.5 ⇒ 0.25 / N
        N = len(aug_dataset)
        wd = 0.25 / max(N, 1)

        train_acc, train_loss = train_one_round(
            model, train_loader, CFG.device,
            epochs=CFG.epochs_per_round, lr=CFG.lr, weight_decay=wd
        )

        # AUC with MC-dropout T=20
        auc = evaluate_auc_mc(model, test_loader, CFG.device, T=CFG.mc_passes)
        acc = evaluate(model, test_loader, CFG.device)

        history["round"].append(r)
        history["labeled"].append(len(labeled_idx))
        history["auc"].append(auc)
        history["acc"].append(acc)

        print(f"AUC={auc:.4f}  ACC={acc:.4f}  N_aug={N}  wd={wd:.6f}")

        if r == CFG.acquisition_rounds - 1 or not unlabeled_idx:
            break

        # acquire k by BALD / entropy / random
        picked, unlabeled_idx = acquire_once(
            model, train_full, unlabeled_idx,
            device=CFG.device, T=CFG.mc_passes, k=CFG.acquisition_size,
            score_subset=CFG.score_subset, test_batch=CFG.test_batch_size,
            acquisition=CFG.acq
        )

        # count positives acquired
        pos_acq = sum(train_full.labels[i] == 1 for i in picked)
        history["pos_acquired"].append(pos_acq)

        labeled_idx += picked
        print(f"acquired {len(picked)} (pos {pos_acq}) → labeled={len(labeled_idx)}  pool={len(unlabeled_idx)}")

        save_ckpt(
            CFG.ckpt_path,
            model=model,
            history=history,
            round_idx=r,
            labeled_idx=labeled_idx,
            unlabeled_idx=unlabeled_idx,
            val_idx=0,
            phase="training"
        )

    # final save
    save_ckpt(
        CFG.ckpt_path,
        model=model,
        history=history,
        round_idx=r,
        labeled_idx=labeled_idx,
        unlabeled_idx=unlabeled_idx,
        val_idx=0,
        phase="training"
    )

    return history

def main():

    acquisition_rounds = 4
    num_runs = 3          # <--- how many independent runs you want

    train_img_dir = os.path.join(ISIC_BASE, "ISBI2016_ISIC_Part3B_Training_Data")
    train_csv = os.path.join(ISIC_BASE, "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv")

    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    train_full = ISICSimpleDataset(train_img_dir, train_csv, transform=train_tf)

    clean_cuda()
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Memory allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
        print("Memory reserved  (MB):", torch.cuda.memory_reserved(0) / 1024**2)

    # base config, just to define seed / other defaults
    base_CFG = Config()
    set_seed(base_CFG.seed)

    # --- FIXED TEST SPLIT (used for all runs & methods) ---
    labeled_idx0, unlabeled_idx0, test_idx = make_balanced_test_and_initial(
        train_full, seed=base_CFG.seed
    )

    print(f"Initial labeled={len(labeled_idx0)}  pool={len(unlabeled_idx0)}  test={len(test_idx)}")

    acq_methods = ["random", "bald"]  # or whatever you want

    # now store a LIST of histories per acquisition method
    all_histories = {acq: [] for acq in acq_methods}

    for run in range(num_runs):
        print("\n" + "=" * 80)
        print(f"Starting run {run+1}/{num_runs} (same test set, new random seed)")
        print("=" * 80)

        # optional: jitter seed per run so MC dropout etc. differ
        run_seed = base_CFG.seed + run
        set_seed(run_seed)

        for acq in acq_methods:
            print("\n" + "#" * 80)
            print(f"Run {run+1}/{num_runs} - acquisition strategy: {acq}")
            print("#" * 80)

            CFG = Config(acq=acq, acquisition_rounds=acquisition_rounds)
            CFG.seed = run_seed
            set_seed(CFG.seed)

            # IMPORTANT: start each run/method from SAME initial labeled/pool split
            labeled_idx = list(labeled_idx0)
            unlabeled_idx = list(unlabeled_idx0)

            history = run_active_learning_ISIC(
                train_full=train_full,
                labeled_idx=labeled_idx,
                unlabeled_idx=unlabeled_idx,
                test_idx=test_idx,   # same test set every time
                CFG=CFG
            )

            # tag which run this history came from (handy when plotting)
            history["run"] = run

            all_histories[acq].append(history)

    # save list-of-histories structure
    with open("./active_learning/ISIC_all_histories.pkl", "wb") as f:
        pickle.dump(all_histories, f)


# def main():

#     acquisition_rounds = 4

#     train_img_dir = os.path.join(ISIC_BASE, "ISBI2016_ISIC_Part3B_Training_Data")
#     train_csv = os.path.join(ISIC_BASE, "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv")

#     train_tf = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406],
#                     [0.229, 0.224, 0.225]),
#     ])

#     train_full = ISICSimpleDataset(train_img_dir, train_csv, transform=train_tf)

#     clean_cuda()
#     print("GPU available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("GPU name:", torch.cuda.get_device_name(0))
#         print("Memory allocated (MB):", torch.cuda.memory_allocated(0) / 1024**2)
#         print("Memory reserved  (MB):", torch.cuda.memory_reserved(0) / 1024**2)

#     # base config, just to define seed / other defaults
#     base_CFG = Config()
#     set_seed(base_CFG.seed)

#     # do a single balanced split once so all acquisition methods share the same test set
#     labeled_idx0, unlabeled_idx0, test_idx = make_balanced_test_and_initial(
#         train_full, seed=base_CFG.seed
#     )

#     print(f"Initial labeled={len(labeled_idx0)}  pool={len(unlabeled_idx0)}  test={len(test_idx)}")

#     acq_methods = ["random","bald"]  # or whatever you want
#     all_histories = {}

#     for acq in acq_methods:
#         print("\n" + "#" * 80)
#         print(f"Running experiment with acquisition strategy: {acq}")
#         print("#" * 80)

#         CFG = Config(acq=acq, acquisition_rounds=acquisition_rounds)

#         labeled_idx = list(labeled_idx0)
#         unlabeled_idx = list(unlabeled_idx0)

#         history = run_active_learning_ISIC(
#             train_full=train_full,
#             labeled_idx=labeled_idx,
#             unlabeled_idx=unlabeled_idx,
#             test_idx=test_idx,
#             CFG=CFG
#         )

#         all_histories[acq] = history

#     with open("./active_learning/ISIC_all_histories.pkl", "wb") as f:
#         pickle.dump(all_histories, f)


if __name__ == "__main__":
    main()