import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchmetrics

from model_common import (
    NUM_CLASSES, BACKGROUND_CLASS, PALETTE, index_to_rgb,
    build_model, compute_class_weights, build_losses, PairedDamageDataset,
)


def make_dataset(split, train_augment):
    return PairedDamageDataset(
        root_dir=DATA_DIR, split=split,
        pre_img_dir="before_aligned", post_img_dir="after_reference",
        aux_mask_dir="semantic_masks_before", damage_mask_dir="semantic_masks_after",
        train_augment=train_augment,
    )


# --- Hyperparameters ---
DATA_DIR = "./input"
BATCH_SIZE = 8
EPOCHS = 300
LR = 1e-4
# NEW: path to xBD-pretrained weights from pretrain_xbd.py. Set to None to train
# from ImageNet-only initialization (the original from-scratch behavior).
PRETRAINED_WEIGHTS = "siam_unet_xbd_pretrained.pth"
# NEW: when loading xBD weights, the encoder AND decoder are now both meaningfully
# pretrained (not just the ImageNet backbone), so a single lower LR multiplier for
# the encoder is less necessary than in the from-scratch case — but keeping a mild
# split still protects the backbone slightly more than the newer/less-trained parts.
ENCODER_LR_MULT = 0.5 if PRETRAINED_WEIGHTS else 0.1
WEIGHT_DECAY = 1e-2
AUX_LOSS_WEIGHT = 0.4
EARLY_STOP_PATIENCE = 40
LR_SCHED_PATIENCE = 15
VIS_EVERY_N_EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    train_dataset = make_dataset("train", train_augment=True)
    val_dataset = make_dataset("valid", train_augment=False)
    test_dataset = make_dataset("test", train_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    torch.set_default_device(DEVICE)
    model = build_model(encoder_name="resnet18", encoder_weights="imagenet")
    torch.set_default_device('cpu')
    model = model.to(DEVICE)

    # NEW: load xBD-pretrained weights if available. strict=True will loudly fail if
    # the architectures have drifted apart (e.g. NUM_CLASSES changed between the two
    # scripts) rather than silently loading a mismatched model.
    if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
        state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded xBD-pretrained weights from {PRETRAINED_WEIGHTS}")
    elif PRETRAINED_WEIGHTS:
        print(f"WARNING: {PRETRAINED_WEIGHTS} not found — training from ImageNet init only.")

    class_weights = compute_class_weights(
        train_dataset.mask_dir, train_dataset.images, NUM_CLASSES, DEVICE
    )
    damage_criterion, aux_criterion, aux_loss_weight = build_losses(
        class_weights, DEVICE, aux_loss_weight=AUX_LOSS_WEIGHT
    )

    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR * ENCODER_LR_MULT},
        {'params': other_params, 'lr': LR},
    ], weight_decay=WEIGHT_DECAY)

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

    log_path = "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_f1", "val_iou", "val_aux_iou", "lr"]
        )

    def save_visuals(loader, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            for batch_idx, (pre, post, aux_target, mask) in enumerate(loader):
                pre, post, aux_target, mask = (
                    pre.to(DEVICE), post.to(DEVICE), aux_target.to(DEVICE), mask.to(DEVICE)
                )
                damage_logits, aux_logits = model(pre, post)
                preds = torch.argmax(damage_logits, dim=1)
                aux_prob = torch.sigmoid(aux_logits)

                save_image(index_to_rgb(preds, PALETTE, DEVICE), f"{out_dir}/pred_batch{batch_idx}.png")
                save_image(index_to_rgb(mask, PALETTE, DEVICE), f"{out_dir}/mask_batch{batch_idx}.png")
                save_image(aux_prob, f"{out_dir}/aux_pred_batch{batch_idx}.png")
                save_image(aux_target, f"{out_dir}/aux_target_batch{batch_idx}.png")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for pre, post, aux_target, mask in train_loader:
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
        current_lr = optimizer.param_groups[-1]['lr']

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
            torch.save(model.state_dict(), "siam_unet_best.pth")
            print(f"--> New best model saved! (IoU: {best_val_iou:.4f})")
            save_visuals(val_loader, "val_outputs/best")
        else:
            epochs_no_improve += 1

        if epoch % VIS_EVERY_N_EPOCHS == 0:
            save_visuals(val_loader, f"val_outputs/epoch{epoch}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nNo improvement in val IoU for {EARLY_STOP_PATIENCE} epochs — stopping early at epoch {epoch+1}.")
            break

    torch.save(model.state_dict(), "siam_unet_last.pth")
    print("Training complete!")

    print("\nStarting Evaluation on Test Split...")
    model.load_state_dict(torch.load("siam_unet_best.pth"))
    model.eval()

    test_loss = 0.0
    f1_metric.reset()
    iou_metric.reset()
    per_class_iou_metric.reset()
    aux_iou_metric.reset()

    with torch.no_grad():
        for pre, post, aux_target, mask in test_loader:
            pre, post, aux_target, mask = (
                pre.to(DEVICE), post.to(DEVICE), aux_target.to(DEVICE), mask.to(DEVICE)
            )

            # aux_target is only used here to report a diagnostic localization score.
            # It is NOT passed into model(pre, post) — the model localizes buildings
            # itself, exactly as it would on a brand-new, unlabeled site.
            damage_logits, aux_logits = model(pre, post)
            loss = damage_criterion(damage_logits, mask) + aux_loss_weight * aux_criterion(aux_logits, aux_target)
            test_loss += loss.item()

            preds = torch.argmax(damage_logits, dim=1)
            f1_metric.update(preds, mask)
            iou_metric.update(preds, mask)
            per_class_iou_metric.update(preds, mask)

            aux_preds = (torch.sigmoid(aux_logits) > 0.5).long().squeeze(1)
            aux_iou_metric.update(aux_preds, aux_target.long().squeeze(1))

        avg_test_loss = test_loss / len(test_loader)
        test_f1 = f1_metric.compute()
        test_iou = iou_metric.compute()
        test_per_class_iou = per_class_iou_metric.compute()
        test_aux_iou = aux_iou_metric.compute()

    print(f"Test Results | Loss: {avg_test_loss:.4f} | F1: {test_f1:.4f} | IoU: {test_iou:.4f} "
          f"| Aux(loc) IoU: {test_aux_iou:.4f}")
    print(f"Test Per-class IoU: {[round(v, 3) for v in test_per_class_iou.tolist()]}")
