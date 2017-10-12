import os
from glob import glob
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Scale, CenterCrop


class SuperResolutionDataset(Dataset):
    def __init__(self, root, large_size=(288, 288), small_size=(72, 72), transforms=None, limit=None):
        self.paths = glob(os.path.join(root, '*'))
        self.transforms = transforms
        self.large_size = large_size
        self.small_size = small_size
        if limit:
            self.paths = self.paths[:limit]

    def make_pair(self, target):
        resize = Compose([Scale(self.large_size[0]),
                          CenterCrop(self.large_size)])
        target = resize(target)
        img = target.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.resize(self.small_size, Image.BILINEAR)
        return img, target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        target = Image.open(self.paths[idx]).convert('RGB')
        img, target = self.make_pair(target)

        if self.transforms:
            # Currently random transforms do not get applied in the same way to both images
            img = self.transforms(img)
            target = self.transforms(target)

        return img, target
