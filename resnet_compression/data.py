"""CIFAR-100 data loading."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import NUM_CLASSES, Config
from .utils import log_section

logger = logging.getLogger(__name__)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Sizes of the synthetic datasets used by ``Config.smoke_test``.
SMOKE_TRAIN_SIZE = 256
SMOKE_TEST_SIZE = 128


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the ``(train, test)`` transform pipelines."""
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Create the CIFAR-100 train and test loaders.

    In smoke-test mode random synthetic images are used instead, so nothing is
    downloaded.
    """
    log_section("Loading CIFAR-100" + (" (synthetic smoke-test data)" if cfg.smoke_test else ""))
    train_transform, test_transform = build_transforms()

    if cfg.smoke_test:
        image_size = (3, 32, 32)
        train_set = datasets.FakeData(
            SMOKE_TRAIN_SIZE, image_size, NUM_CLASSES, transform=train_transform, random_offset=0
        )
        test_set = datasets.FakeData(
            SMOKE_TEST_SIZE,
            image_size,
            NUM_CLASSES,
            transform=test_transform,
            random_offset=SMOKE_TRAIN_SIZE,
        )
    else:
        train_set = datasets.CIFAR100(
            cfg.data_dir, train=True, download=True, transform=train_transform
        )
        test_set = datasets.CIFAR100(
            cfg.data_dir, train=False, download=True, transform=test_transform
        )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    logger.info("Training samples: %d", len(train_set))
    logger.info("Test samples: %d", len(test_set))
    logger.info("Classes: %d", NUM_CLASSES)
    return train_loader, test_loader
