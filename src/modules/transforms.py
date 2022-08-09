from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    ShiftScaleRotate,
    HueSaturationValue,
    RandomContrast,
    RandomBrightness,
    OneOf,
    Compose
)

from albumentations.pytorch import ToTensor

train_transform = [
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(rotate_limit=(-30,30), p=0.5),
    HueSaturationValue(p=0.5),
    RandomContrast(p=0.5),
    RandomBrightness(p=0.5),
    Resize(416, 416),
    ToTensor(),
]


val_transform = [
    Resize(416, 416),
    ToTensor(),
]



def get_transforms(config, mode):
    list_transforms = []
    config_tarnsform = config[mode]["transform"]
    if config_tarnsform["HorizontalFlip"]:
        list_transforms.append(HorizontalFlip())
    if config_tarnsform["VerticalFlip"]:
        list_transforms.append(VerticalFlip())
    if config_tarnsform["Contrast"]:
        list_transforms.append(
            OneOf([
                RandomContrast(0.5),
                RandomBrightness(),
            ], p=0.5),
        )
    if config_tarnsform["ShiftScaleRotate"]:
        list_transforms.append(ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.05, 0.05), rotate_limit=(-30,30), p=0.5))

    list_transforms.extend(
        [
            Resize(config["model"]["image_size"], config["model"]["image_size"]),
            # Normalize(mean=phase_config.mean, std=phase_config.std, p=1),
            ToTensor(),
        ]
    )

    return list_transforms
