import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
import numpy as np

# Loader for .npy vector files
def _npy_loader(path: str) -> np.ndarray:
    return np.load(path)

class SketchDataset(Dataset):
    """
    A unified SketchDataset that wraps:
      - an ImageFolder for raster images
      - a DatasetFolder for vector (.npy) data
    mode: 'raster' or 'vector'
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        mode: str = 'raster',
        raster_transform=None,
        vector_transform=None,
        vector_dir: str = None
    ):
        self.mode = mode
        self.split = split
        self.root_dir = root_dir

        if self.mode == 'raster':
            data_path = os.path.join(self.root_dir, 'quickdraw_raster', self.split)
            self.dataset = ImageFolder(
                root=data_path,
                transform=raster_transform
            )
        elif self.mode == 'vector':
            base_dir = vector_dir or os.path.join(self.root_dir, 'quickdraw_vector')
            data_path = os.path.join(base_dir, self.split)
            self.dataset = DatasetFolder(
                root=data_path,
                loader=_npy_loader,
                extensions=['.npy'],
                transform=vector_transform
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        # for consistency with ImageFolder API
        return self.dataset.class_to_idx

# Example usage:
# from dataset import SketchDataset
#
# raster_ds = SketchDataset(
#     root_dir=PROJECT_ROOT,
#     split='train',
#     mode='raster',
#     raster_transform=data_transforms['train']
# )
# raster_loader = DataLoader(raster_ds, batch_size=..., shuffle=True, ...)
#
# vector_ds = SketchDataset(
#     root_dir=PROJECT_ROOT,
#     split='train',
#     mode='vector',
#     vector_transform=my_vector_transform
# )
# vector_loader = DataLoader(vector_ds, ...)
