from torchvision import datasets
import torchvision.transforms as T
import torch
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

from cifar.base import BaseDataLoader


MEANS = [0.5071, 0.4867, 0.4408]
STDS  = [0.2675, 0.2565, 0.2761]


class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = T.Compose([
            T.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training,
                                        download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar100DataLoader(BaseDataLoader):
    """
    CIFAR10 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEANS, std=STDS)
        ]) if not training else T.Compose([
            CutoutAugs()
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training,
                                         download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# -- Cutout --
# https://github.com/uoguelph-mlrg/Cutout

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class CutoutAugs:

    augs = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=MEANS, std=STDS),
        Cutout()
    ])

    def __call__(self, img):
        return self.augs(img)


# -- generic augs --

class LightAugs:

    augs = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip()
    ])

    def __call__(self, img):
        return self.augs(img)


class MediumAugs:

    augs = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Sometimes(
            0.5,
            iaa.Sequential([
                iaa.Affine(
                    scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    rotate=(-10, 10),
                    order=[0, 1],
                    mode=ia.ALL
                )
            ])
        ),
    ])

    def __call__(self, img):
        return self.augs.augment_image(np.array(img))


class HeavyAugs:
    """
    https://imgaug.readthedocs.io/en/latest/source/examples_basics.html
    """

    augs = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    def __call__(self, img):
        return self.augs.augment_image(np.array(img))