import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

from utils import _get_value

## DATASET
TRAIN_SET = datasets.MNIST(root='data', train=True, download=True)
TEST_SET = datasets.MNIST(root='data', train=False, download=True)

HOLDOUT_CLASSES = {0: [],  # top_left
                   1: [1, 2, 3],  # top_right
                   2: [4, 5, 6],  # bottom_left
                   3: [7, 8, 9, 0]}  # bottom_right

TRANSLATION_FACTOR = (40 - 28) / 2 / 40
TRANSFORM = transforms.Compose([
    transforms.CenterCrop([40, 40]),  # basically pads with zeros
    transforms.RandomAffine(0, translate=[TRANSLATION_FACTOR, TRANSLATION_FACTOR]),
    transforms.ToTensor()
])

class BlockStylePVR(Dataset):
    def __init__(self,
                 train: bool,
                 mode: str = "iid",
                 size: int = None):
        """

        Args:
            train: whether to use MNIST train set, else use the test set.
            mode: "holdout" or "adversarial" or "iid".
            size: dataset size
        """

        self.ds = TRAIN_SET if train else TEST_SET
        #size train is 60.000
        if size is not None and size > len(self.ds) // 4:
            raise ValueError(f"Requested dataset size is too big. Can be up too {len(self.ds) // 4}.")

        # use maximum size if size is null
        self.pvr_ds_size = len(self.ds) // 4 if size is None else int(size)

        # the labels (0-9) of the images in the dataset
        ds_labels = []
        for idx, (c, y) in enumerate(self.ds): #idx index, _ image PIL, y label
            ds_labels.append(y)
        ds_labels = torch.tensor(ds_labels)

        # dtype must be torch.long, otherwise it will crash during training
        self.idxs = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)
        self.labels = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)  # 0-9

        if mode == 'iid':
            # the labels (0-9) of the 4 digits in each sample
            self.labels = ds_labels.reshape([-1, 4])  # group into 4's
            self.labels = self.labels[:self.pvr_ds_size]  # trim to requested size

            # the original idx of the 4 digits in each sample
            self.idxs = torch.arange(len(self.ds)).reshape([-1, 4])  # group into 4's
            self.idxs = self.idxs[:self.pvr_ds_size]  # trim to requested size

        elif mode == "holdout":
            # sample from the ds excluding the labels that are held out
            for i, holdout_class in HOLDOUT_CLASSES.items():
                probs = torch.ones(len(self.ds))
                for label in holdout_class:
                    probs[ds_labels == label] = 0
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = ds_labels[curr_idxs]

        elif mode == "adversarial":
            # sample from the ds only where the labels are held out
            for i, holdout_class in HOLDOUT_CLASSES.items():
                probs = torch.ones(len(self.ds)) if i == 0 else torch.zeros(len(self.ds))
                for label in holdout_class:
                    probs[ds_labels == label] = 1
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = ds_labels[curr_idxs]

        else:
            raise ValueError("Unknown dataset mode.")

    def __getitem__(self, idx):
        labels = self.labels[idx]
        idxs = self.idxs[idx]

        # for each label, get the matching image from the ds using the idx
        # transform it (fit into 40x40 and translate randomly)
        # and then put in the appropriate location in the result image
        x = torch.zeros([1, 80, 80])
        x[0, :40, :40] = TRANSFORM(self.ds[idxs[0]][0])
        x[0, :40, 40:] = TRANSFORM(self.ds[idxs[1]][0])
        x[0, 40:, :40] = TRANSFORM(self.ds[idxs[2]][0])
        x[0, 40:, 40:] = TRANSFORM(self.ds[idxs[3]][0])

        # calculate the value based on the pointer
        value = _get_value(labels)

        return x, value

    def __len__(self):
        return self.pvr_ds_size


def setup_loaders():
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    AVAIL_CPUS = multiprocessing.cpu_count()
    torch.manual_seed(42)

    num_workers = (4 * AVAIL_GPUS) if (AVAIL_GPUS > 0) else AVAIL_CPUS

    shuffle = False
    # Aumentar size para dataset completo
    train_ds = BlockStylePVR(train=True, size=100)
    train_loader = DataLoader(train_ds, batch_size=2,
                              pin_memory=True,
                              num_workers=num_workers,
                              shuffle=shuffle)

    val_ds = BlockStylePVR(train=False, size=100)
    val_loader = DataLoader(val_ds, batch_size=2,
                            pin_memory=True,
                            num_workers=num_workers)

    return train_loader, val_loader
