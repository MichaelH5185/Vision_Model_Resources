import os
import csv
import numpy as np
import random 
import shutil
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchmetrics

from model_common import (
    NUM_CLASSES, BACKGROUND_CLASS, PALETTE, index_to_rgb,
    build_model, compute_class_weights, build_losses, PairedDamageDataset,
)

# --- xBD layout ---
XBD_ROOT = "./dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "test"   # set to "hold" instead if your download uses that split name
PRE_IMG_DIR = "pre-images"
PRE_MASK_DIR = "pre-masks"
POST_IMG_DIR = "post-images"
POST_MASK_DIR = "post-masks"

VAL_FRACTION = 0.15
SPLIT_SEED = 42
MOVE_FILES = True
PRE_SUFFIX = "_pre_disaster"    # adjust if your naming differs
POST_SUFFIX = "_post_disaster"
 
 
def _index_by_id(dir_path, suffix):
    """Map canonical id -> actual filename in dir_path, for filenames shaped like
    '<id><suffix><ext>' (e.g. 'guatemala-volcano_00000001_pre_disaster.png' -> id
    'guatemala-volcano_00000001'). Handles per-folder extensions independently."""
    mapping = {}
    for fname in os.listdir(dir_path):
        stem, ext = os.path.splitext(fname)
        if stem.endswith(suffix):
            _id = stem[: -len(suffix)]
            mapping[_id] = fname
    return mapping
 
 
def undo_partial_split(root_dir, dirs):
    """One-time recovery: if an earlier buggy run moved some folders into train/test
    while leaving others flat (mismatched filenames across folders), this moves
    everything back up to the flat layout so create_train_test_split can be re-run
    cleanly. Not called automatically — run it once yourself if you hit that state,
    then remove/comment out the call."""
    for split in ("train", "test"):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for d in dirs:
            src_dir = os.path.join(split_dir, d)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(root_dir, d)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
        shutil.rmtree(split_dir)
    print("Reverted partial split back to flat layout.")
 
 
def create_train_test_split(root_dir, pre_img_dir, post_img_dir, pre_mask_dir, post_mask_dir,
                             pre_suffix=PRE_SUFFIX, post_suffix=POST_SUFFIX,
                             val_frac=VAL_FRACTION, seed=SPLIT_SEED, move=MOVE_FILES):
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        print("Train/test split already exists — skipping split step.")
        return
 
    flat_pre_img_dir = os.path.join(root_dir, pre_img_dir)
    if not os.path.isdir(flat_pre_img_dir):
        raise FileNotFoundError(
            f"Expected a flat folder at {flat_pre_img_dir} to split from, but it doesn't "
            "exist (and train/test aren't already there either). Check XBD_ROOT and the "
            "folder-name constants above."
        )
 
    # Index each folder by its canonical id (suffix stripped), rather than assuming
    # identical filenames across folders — pre/post files share an id but differ by
    # the _pre_disaster / _post_disaster suffix.
    indices = {
        pre_img_dir: (_index_by_id(os.path.join(root_dir, pre_img_dir), pre_suffix), pre_suffix),
        post_img_dir: (_index_by_id(os.path.join(root_dir, post_img_dir), post_suffix), post_suffix),
        pre_mask_dir: (_index_by_id(os.path.join(root_dir, pre_mask_dir), pre_suffix), pre_suffix),
        post_mask_dir: (_index_by_id(os.path.join(root_dir, post_mask_dir), post_suffix), post_suffix),
    }
 
    ids = sorted(indices[pre_img_dir][0].keys())
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
 
    op = shutil.move if move else shutil.copy2
    for split, split_ids in [("train", train_ids), ("test", val_ids)]:
        for d in indices:
            os.makedirs(os.path.join(root_dir, split, d), exist_ok=True)
        for _id in split_ids:
            for d, (index, _) in indices.items():
                if _id not in index:
                    print(f"  WARNING: no file for id={_id} in {d}, skipping")
                    continue
                fname = index[_id]
                ext = os.path.splitext(fname)[1]
                src = os.path.join(root_dir, d, fname)
                # Renamed to '<id><ext>' (suffix stripped) on the way in, so all four
                # folders end up with matching filenames per example — what
                # PairedDamageDataset expects.
                dst = os.path.join(root_dir, split, d, f"{_id}{ext}")
                op(src, dst)
 
    print(f"Split complete: {len(train_ids)} train / {len(val_ids)} test.")
 
 
# One-time recovery from the earlier buggy split (comment this out after your first
# successful run, or leave it — it's a no-op once nothing is left under train/test):
# undo_partial_split(XBD_ROOT, [PRE_IMG_DIR, POST_IMG_DIR, PRE_MASK_DIR, POST_MASK_DIR])
 
#create_train_test_split(XBD_ROOT, PRE_IMG_DIR, POST_IMG_DIR, PRE_MASK_DIR, POST_MASK_DIR)



def inspect_mask_values(mask_dir, image_names, n=5):
    sample = image_names[:n]
    for name in sample:
        arr = np.array(Image.open(os.path.join(mask_dir, name)).convert("L"))
        print(f"  {name}: unique values = {sorted(np.unique(arr).tolist())}")


def make_dataset(split, train_augment):
    return PairedDamageDataset(
        root_dir=XBD_ROOT, split=split,
        pre_img_dir=PRE_IMG_DIR, post_img_dir=POST_IMG_DIR,
        aux_mask_dir=PRE_MASK_DIR, damage_mask_dir=POST_MASK_DIR,
        train_augment=train_augment,
    )


# --- Hyperparameters ---
BATCH_SIZE = 8
EPOCHS = 50                 # xBD is orders of magnitude larger than the drone set;
                             # far fewer epochs are needed and overfitting isn't the
                             # concern here, unlike the fine-tuning run.
LR = 1e-4
WEIGHT_DECAY = 1e-2
AUX_LOSS_WEIGHT = 0.4
EARLY_STOP_PATIENCE = 15     # safety net, not the main defense (dataset is large)
LR_SCHED_PATIENCE = 4
VIS_EVERY_N_EPOCHS = 5
OUTPUT_WEIGHTS = "siam_unet_xbd_pretrained.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    train_dataset = make_dataset(TRAIN_SPLIT, train_augment=True)
    val_dataset = make_dataset(VAL_SPLIT, train_augment=False)
    print(f"xBD train pairs: {len(train_dataset)} | val pairs: {len(val_dataset)}")

    print("Sample post-mask (damage) values — expect a subset of [0,1,2,3,4,5]:")
    inspect_mask_values(train_dataset.mask_dir, train_dataset.images)
    print("Sample pre-mask (aux/footprint) values — expect [0] and/or [0,255] or [0,1]:")
    inspect_mask_values(train_dataset.aux_mask_dir, train_dataset.images)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    torch.set_default_device(DEVICE)
    model = build_model(encoder_name="timm-efficientnet-b0", encoder_weights="noisy-student")
    torch.set_default_device('cpu')
    model = model.to(DEVICE)

    class_weights = compute_class_weights(
        train_dataset.mask_dir, train_dataset.images, NUM_CLASSES, DEVICE
    )
    damage_criterion, aux_criterion, aux_loss_weight = build_losses(
        class_weights, DEVICE, aux_loss_weight=AUX_LOSS_WEIGHT
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=LR_SCHED_PATIENCE
    )

    f1_metric = torchmetrics.classification.MulticlassF1Score(
        num_classes=NUM_CLASSES, average="macro", ignore_index=BACKGROUND_CLASS
    ).to(DEVICE)
    iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average="macro", ignore_index=BACKGROUND_CLASS
    ).to(DEVICE)
    per_class_iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
        num_classes=NUM_CLASSES, average=None
    ).to(DEVICE)
    aux_iou_metric = torchmetrics.classification.BinaryJaccardIndex().to(DEVICE)

    best_val_iou = 0.0
    epochs_no_improve = 0

    log_path = "pretrain_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_f1", "val_iou", "val_aux_iou", "lr"]
        )

    def save_visuals(loader, out_dir, max_batches=3):
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            for batch_idx, (pre, post, aux_target, mask) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                pre, post, aux_target, mask = (
                    pre.to(DEVICE), post.to(DEVICE), aux_target.to(DEVICE), mask.to(DEVICE)
                )
                damage_logits, aux_logits = model(pre, post)
                preds = torch.argmax(damage_logits, dim=1)
                save_image(index_to_rgb(preds, PALETTE, DEVICE), f"{out_dir}/pred_batch{batch_idx}.png")
                save_image(index_to_rgb(mask, PALETTE, DEVICE), f"{out_dir}/mask_batch{batch_idx}.png")
                save_image(torch.sigmoid(aux_logits), f"{out_dir}/aux_pred_batch{batch_idx}.png")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for step, (pre, post, aux_target, mask) in enumerate(train_loader):
            pre, post, aux_target, mask = (
                pre.to(DEVICE), post.to(DEVICE), aux_target.to(DEVICE), mask.to(DEVICE)
            )

            optimizer.zero_grad()
            damage_logits, aux_logits = model(pre, post)
            loss = damage_criterion(damage_logits, mask) + aux_loss_weight * aux_criterion(aux_logits, aux_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        f1_metric.reset()
        iou_metric.reset()
        per_class_iou_metric.reset()
        aux_iou_metric.reset()

        with torch.no_grad():
            for pre, post, aux_target, mask in val_loader:
                pre, post, aux_target, mask = (
                    pre.to(DEVICE), post.to(DEVICE), aux_target.to(DEVICE), mask.to(DEVICE)
                )

                damage_logits, aux_logits = model(pre, post)
                loss = damage_criterion(damage_logits, mask) + aux_loss_weight * aux_criterion(aux_logits, aux_target)
                val_loss += loss.item()

                preds = torch.argmax(damage_logits, dim=1)
                f1_metric.update(preds, mask)
                iou_metric.update(preds, mask)
                per_class_iou_metric.update(preds, mask)

                aux_preds = (torch.sigmoid(aux_logits) > 0.5).long().squeeze(1)
                aux_iou_metric.update(aux_preds, aux_target.long().squeeze(1))

            avg_val_loss = val_loss / len(val_loader)

        val_f1 = f1_metric.compute()
        val_iou = iou_metric.compute()
        per_class_iou = per_class_iou_metric.compute()
        val_aux_iou = aux_iou_metric.compute()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"| Val F1: {val_f1:.4f} | Val IoU: {val_iou:.4f} | Val Aux(loc) IoU: {val_aux_iou:.4f} | LR: {current_lr:.2e}")
        print(f"    Per-class IoU: {[round(v, 3) for v in per_class_iou.tolist()]}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_train_loss, avg_val_loss,
                                     val_f1.item(), val_iou.item(), val_aux_iou.item(), current_lr])

        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"--> New best model saved! (IoU: {best_val_iou:.4f})")
        else:
            epochs_no_improve += 1

        if epoch % VIS_EVERY_N_EPOCHS == 0:
            save_visuals(val_loader, f"pretrain_val_outputs/epoch{epoch}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nNo improvement in val IoU for {EARLY_STOP_PATIENCE} epochs — stopping early at epoch {epoch+1}.")
            break

    print(f"\nPretraining complete. Best weights saved to {OUTPUT_WEIGHTS}")
    print("Copy or symlink this file next to training96_aug.py (or update PRETRAINED_WEIGHTS "
          "in that script) to fine-tune on your drone dataset.")
