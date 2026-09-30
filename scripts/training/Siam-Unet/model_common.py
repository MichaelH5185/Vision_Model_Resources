"""
Shared model + utility code for the xBD pretraining script (pretrain_xbd.py) and the
drone fine-tuning script (training96_aug.py).

Keeping the model definition in one place is deliberate: the two scripts must build
the *exact* same architecture (same encoder, same decoder config, same aux head) or
`model.load_state_dict(...)` in the fine-tuning script will fail or silently mismatch.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import change_detection_pytorch as cdp

# Class 0 = background/no-building for the 7-class damage taxonomy used throughout.
# 1=no-damage(green) 2=minor(yellow) 3=major(orange) 4=destroyed(red) 5=un-classified
# (magenta) 6=<custom class specific to your drone taxonomy, unused during xBD
# pretraining>. Adjust here if your actual class mapping differs.
NUM_CLASSES = 7
BACKGROUND_CLASS = 0

PALETTE = torch.tensor([
    [0, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [255, 165, 0],
    [255, 0, 0],
    [255, 0, 255],
    [0, 255, 255]
], dtype=torch.float32)


def index_to_rgb(mask_idx, palette, device):
    palette = palette.to(device)
    rgb = palette[mask_idx]
    rgb = rgb.permute(0, 3, 1, 2) / 255.0
    return rgb


class AuxLocalizationHead(nn.Module):
    """Small U-Net-style decoder that predicts a 1-channel building-footprint logit
    map from an encoder's multi-scale feature pyramid, independent of cdp's main
    siamese decoder. See training96_aug.py for the rationale (replaces feeding a
    ground-truth building mask directly into the model's input)."""

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()
        enc_ch = list(encoder_channels[1:])[::-1]  # drop raw-input stage, deep -> shallow
        # Adapt decoder_channels to the encoder's actual depth instead of silently
        # mismatching (via zip() truncation) when an encoder doesn't have exactly
        # 5 stages. Extend by repeating the smallest channel count if the encoder
        # is deeper than the default; truncate if it's shallower.
        n_stages = len(enc_ch)
        if n_stages > len(decoder_channels):
            decoder_channels = tuple(decoder_channels) + (decoder_channels[-1],) * (n_stages - len(decoder_channels))
        else:
            decoder_channels = tuple(decoder_channels[:n_stages])
        in_ch = [enc_ch[0]] + list(decoder_channels[:-1])
        skip_ch = list(enc_ch[1:]) + [0]

        blocks = []
        for i_ch, s_ch, o_ch in zip(in_ch, skip_ch, decoder_channels):
            blocks.append(nn.Sequential(
                nn.Conv2d(i_ch + s_ch, o_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(o_ch),
                nn.ReLU(inplace=True),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def forward(self, features):
        feats = list(features[1:])[::-1]
        x = feats[0]
        skips = feats[1:]
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)


class SiamUnetWithAuxLocalization(cdp.Unet):
    """cdp.Unet plus an auxiliary localization head fed by the pre-disaster encoder
    features. forward() returns (damage_logits, aux_localization_logits). Used
    identically by both the xBD pretraining run and the drone fine-tuning run."""

    def __init__(self, *args, aux_decoder_channels=(256, 128, 64, 32, 16), **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_head = AuxLocalizationHead(self.encoder.out_channels, aux_decoder_channels)

    def forward(self, x1, x2):
        if self.siam_encoder:
            pre_features = self.encoder(x1)
            post_features = self.encoder(x2)
        else:
            pre_features = self.encoder(x1)
            post_features = self.encoder_non_siam(x2)

        decoder_output = self.decoder(pre_features, post_features)
        damage_logits = self.segmentation_head(decoder_output)
        aux_logits = self.aux_head(pre_features)

        return damage_logits, aux_logits


def build_model(encoder_name="resnet18", encoder_weights="imagenet"):
    """Single source of truth for model construction, used by both scripts."""
    return SiamUnetWithAuxLocalization(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        siam_encoder=True,
        fusion_form="diff",
    )


def compute_class_weights(mask_dir, image_names, num_classes, device, mask_ext=None):
    """Inverse-pixel-frequency class weights for CrossEntropyLoss. `mask_ext`, if
    given, replaces each name's extension (useful when image and mask files share a
    stem but differ in extension, e.g. xBD's .png images vs. cached .png masks)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for name in image_names:
        fname = name if mask_ext is None else os.path.splitext(name)[0] + mask_ext
        m = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L"))
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    print("Class pixel counts:", counts.astype(int).tolist())
    print("Class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


class PairedDamageDataset(Dataset):
    """Generic pre/post disaster dataset: pre-image, post-image, a binary aux mask
    (building footprint, used only as an auxiliary-head training TARGET — never
    concatenated onto the model's input), and a multi-class damage mask. Both the
    drone dataset and the pre-rasterized xBD dataset share this exact shape, just
    with different folder names, so both scripts instantiate this one class rather
    than maintaining separate near-duplicate Dataset code.

    Assumes matching filenames for a given example across all four folders.
    """

    def __init__(self, root_dir, split, pre_img_dir, post_img_dir, aux_mask_dir, damage_mask_dir,
                 img_size=(1024, 1024), train_augment=False):
        self.root_dir = os.path.join(root_dir, split)
        self.pre_dir = os.path.join(self.root_dir, pre_img_dir)
        self.post_dir = os.path.join(self.root_dir, post_img_dir)
        self.aux_mask_dir = os.path.join(self.root_dir, aux_mask_dir)
        self.mask_dir = os.path.join(self.root_dir, damage_mask_dir)

        self.images = sorted(os.listdir(self.pre_dir))

        mask_targets = {'post_image': 'image', 'aux_mask': 'mask', 'damage_mask': 'mask'}

        if train_augment:
            # Geometric transforms stay shared across pre, post, and both masks so
            # everything remains spatially aligned.
            self.geo_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets=mask_targets)
            # Photometric jitter is sampled INDEPENDENTLY for pre vs. post (see
            # __getitem__) so the model practices on realistic lighting/sensor drift
            # between the two captures instead of identical jitter applied to both.
            self.photo_transform = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ])
        else:
            self.geo_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
            ], additional_targets=mask_targets)
            self.photo_transform = None

        self.final_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        pre_np = np.array(Image.open(os.path.join(self.pre_dir, img_name)).convert("RGB"))
        post_np = np.array(Image.open(os.path.join(self.post_dir, img_name)).convert("RGB"))
        aux_np = np.array(Image.open(os.path.join(self.aux_mask_dir, img_name)).convert("L"))
        mask_np = np.array(Image.open(os.path.join(self.mask_dir, img_name)).convert("L"))

        geo = self.geo_transform(
            image=pre_np, post_image=post_np, aux_mask=aux_np, damage_mask=mask_np
        )
        pre_np, post_np = geo['image'], geo['post_image']
        aux_np, mask_np = geo['aux_mask'], geo['damage_mask']

        if self.photo_transform is not None:
            pre_np = self.photo_transform(image=pre_np)['image']
            post_np = self.photo_transform(image=post_np)['image']

        # x1/x2 are plain 3-channel RGB — no building-mask channel appended.
        x1 = self.final_transform(image=pre_np)['image']
        x2 = self.final_transform(image=post_np)['image']

        aux_target = torch.from_numpy(aux_np)
        aux_target = (aux_target > 0).float().unsqueeze(0)

        y = torch.from_numpy(mask_np).long()

        return x1, x2, aux_target, y


def build_losses(class_weights, device, aux_loss_weight=0.4):
    """Returns (damage_criterion, aux_criterion, aux_loss_weight) closures shared by
    both scripts, so the loss formulation used during pretraining matches fine-tuning."""
    ce_loss = nn.CrossEntropyLoss(weight=class_weights).to(device)
    dice_loss = cdp.losses.DiceLoss(mode="multiclass", ignore_index=BACKGROUND_CLASS).to(device)

    def damage_criterion(outputs, mask):
        return ce_loss(outputs, mask) + dice_loss(outputs, mask)

    aux_bce_loss = nn.BCEWithLogitsLoss().to(device)
    aux_dice_loss = cdp.losses.DiceLoss(mode="binary").to(device)

    def aux_criterion(aux_logits, aux_target):
        return aux_bce_loss(aux_logits, aux_target) + aux_dice_loss(aux_logits, aux_target)

    return damage_criterion, aux_criterion, aux_loss_weight
