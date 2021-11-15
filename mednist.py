from glob import glob
import os
import tarfile
from typing import List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MEDNISTDIR = os.path.join(os.path.dirname(__file__), 'data')
MEDNISTCLASSES = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']


def download_mednist(data_dir: str) -> None:
    """
    Download the MNIST dataset from the internet and save it to the given
    directory.
    """
    # Create data_dir if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # download MedNIST
    url = 'https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz'
    filename = 'MedNIST.tar.gz'
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print('Downloading MedNIST...')
        os.system(f'curl -L -o {filepath} {url}')

    if not os.path.exists(os.path.join(data_dir, "MedNIST")):
        print(f"Extracting {filename}...")
        tarfile.open(filepath, "r:gz").extractall(data_dir)


def get_mednist_files(data_dir: str, classname: str) -> List:
    """
    Get a list of all files in the given class.
    """
    return glob(os.path.join(data_dir, 'MedNIST', classname, '*.jpeg'))


def get_normal_files(data_dir: str, normal_class: str) -> List:
    """
    Get a list of all files considered normal.
    """
    return get_mednist_files(data_dir, normal_class)


def get_anomal_files(data_dir: str, normal_class: str) -> List:
    """
    Get a list of files that don't belong to the normal class.
    """
    files = []
    for classname in MEDNISTCLASSES:
        if classname != normal_class:
            files += get_mednist_files(data_dir, classname)
    return files


class MedNISTDataset(Dataset):
    """
    Dataset class for the MedNIST dataset.
    """

    def __init__(self, files: List):
        """
        :param files: List of files to load.
        """
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        :param idx: Index of the file to load.
        :return: The loaded image
        """
        image = Image.open(self.files[idx])
        image = image.convert('L')
        image = transforms.ToTensor()(image)
        return image



class MedNISTTestDataset(Dataset):
    """
    Dataset class for the MedNIST dataset.
    """

    def __init__(self, files: List, labels: List):
        """
        :param files: List of files to load.
        :param labels: List of labels for each file.
        """
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        :param idx: Index of the file to load.
        :return: A tuple containing the loaded image and its label.
        """
        image = Image.open(self.files[idx])
        image = image.convert('L')
        image = transforms.ToTensor()(image)
        label = self.labels[idx]
        return (image, label)


if __name__ == '__main__':
    # download the dataset
    download_mednist(MEDNISTDIR)

    # Get headCT and Hand files
    head_ct_files = get_mednist_files(MEDNISTDIR, 'HeadCT')
    hand_files = get_mednist_files(MEDNISTDIR, 'Hand')
    print(len(head_ct_files), len(hand_files))

    # Create dataset
    ds = MedNISTDataset(head_ct_files)

    # Get a sample and print statistics
    x = next(iter(ds))
    print(x.min(), x.max(), x.shape, x.dtype, x.mean(), x.std())
