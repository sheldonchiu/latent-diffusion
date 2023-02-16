import os
from glob import glob
import numpy as np
import PIL
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

image_format = ['jpeg','jpg', 'png', 'webp']


class LocalBase(Dataset):
    def __init__(self,
                 image_paths,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.image_paths = image_paths
        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": [l for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class LocalTrain(LocalBase):
    def __init__(self, **kwargs):
        super().__init__(image_paths=list(chain(*[glob(os.path.join("/mnt/d/data/vae_train", f"*.{f}")) for f in image_format])), **kwargs)


class LocalValidation(LocalBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(image_paths=list(chain(*[glob(os.path.join("/mnt/d/data/vae_val", f"*.{f}")) for f in image_format])),
                         flip_p=flip_p, **kwargs)