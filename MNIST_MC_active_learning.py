from active_learning import *
# MNIST config
class Config:
    def __init__(self, acq = 'random', acquisition_rounds = 5):
        self.acq = acq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training settings
        self.batch_size = 128
        self.test_batch_size = 1024
        self.epochs_per_round = 10
        self.lr = 1e-3
        self.weight_decay = 5e-4

        # AL
        self.initial_labeled = 20
        self.validation_set_size = 100
        self.acquisition_size = 10
        self.acquisition_rounds = acquisition_rounds
        self.mc_passes = 20
        self.score_subset = 5000
        self.seed = 42

        # data
        self.dataset = "mnist"
        self.data_root = "./data"

        # ckpt file unique to strategy
        self.ckpt_path = f"./active_learning/mnist_mc_{acq}.ckpt"


def main():

    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    CFG = Config()

    labeled_idx,val_idx, unlabeled_idx = balanced_initial_split(train_set,CFG.initial_labeled,CFG.validation_set_size)
    print(len(labeled_idx),len(val_idx), len(unlabeled_idx))  # 100, 59900
    print(val_idx)

    acq_methods = ["random","entropy","bald", "var"]
    mc_count = [1, 5, 10, 20]
    all_histories = {}
    acq = 'bald'

    for s in mc_count:
        print('Training for {} acquisition strategy'.format(s))
        CFG = Config(acq = acq, acquisition_rounds=30)
        CFG.mc_passes = s
        CFG.ckpt_path = f"./active_learning/mnist_mc_{s}_{acq}.ckpt"

        set_seed(CFG.seed)
        labeled_idx,val_idx, unlabeled_idx = balanced_initial_split(train_set,CFG.initial_labeled,CFG.validation_set_size)
        model = MNIST_CNN()
        history = active_learning_loop(model, train_set, test_set, labeled_idx, val_idx, unlabeled_idx, CFG,resume = True, ckpt_path = CFG.ckpt_path)
        all_histories[s] = history


    with open("./active_learning/MNIST_all_histories.pkl", "wb") as f:
        pickle.dump(all_histories, f)

if __name__ == "__main__":
    main()