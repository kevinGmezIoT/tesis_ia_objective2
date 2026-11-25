import torchvision.transforms as T


def get_transforms(image_size, is_train=True, color_jitter=True):
    # image_size: (H, W)
    if is_train:
        ops = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(image_size),
        ]
        if color_jitter:
            ops.append(
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)
            )
        ops.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5),
        ])
        transform = T.Compose(ops)
    else:
        transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transform
