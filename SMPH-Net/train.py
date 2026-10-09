"""SMPH-Net single fold training script.

Usage:
    python train.py --train_csv data/fold_0_train.csv --val_csv data/fold_0_val.csv \
        --root_dir data --fold 0 --num_epochs 50 --batch_size 4 --lr 1e-4
"""
import argparse
import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from datasets.cmr_dataset import CMRDataset
from losses.loss import PHLoss
from models.dmf_net import DMFNet
from utils.metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, train_loader, criterion, optimizer,
                    flow_loss_weight=0.1, grad_accum_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred, flow_loss = model(x)
        pred = pred.squeeze(1)
        loss = (criterion(pred, y) + flow_loss_weight * flow_loss) / grad_accum_steps
        loss.backward()
        total_loss += loss.item() * grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    if len(train_loader) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, flow_loss_weight=0.1):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred, flow_loss = model(x)
            pred_for_loss = pred.squeeze(1)
            loss = criterion(pred_for_loss, y) + flow_loss_weight * flow_loss
            total_loss += loss.item()
            preds.append(pred_for_loss)
            labels.append(y)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc, auc, f1 = compute_metrics(preds, labels)
    return total_loss / len(val_loader), acc, auc, f1


def train_fold(fold, train_csv, val_csv, root_dir, num_epochs=50, batch_size=4,
               lr=1e-4, num_frames=16, output_dir='outputs',
               flow_loss_weight=0.1, fusion_type='cross_attention',
               early_stopping_patience=15, num_workers=8, grad_accum_steps=1):
    """Train a single fold."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f'fold_{fold}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Training Fold {fold}")
    logger.info(f"Train: {train_csv}")
    logger.info(f"Val: {val_csv}")
    logger.info(f"Frames: {num_frames}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    train_dataset = CMRDataset(csv_file=train_csv, root_dir=root_dir,
                               num_frames=num_frames)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False,
                              prefetch_factor=2 if num_workers > 0 else None)
    val_dataset = CMRDataset(csv_file=val_csv, root_dir=root_dir,
                             num_frames=num_frames)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=val_dataset.collate_fn,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=2 if num_workers > 0 else None)

    model = DMFNet(fusion_type=fusion_type).to(device)
    criterion = PHLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )

    best_auc = 0
    best_model_path = os.path.join(output_dir, f"best_model_fold_{fold}.pth")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     flow_loss_weight, grad_accum_steps)
        val_loss, acc, auc, f1 = validate(model, val_loader, criterion,
                                          flow_loss_weight)

        logger.info(f"Fold {fold} Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Acc={acc:.4f}, AUC={auc:.4f}, "
                    f"F1={f1:.4f}")

        scheduler.step(auc)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> Best model saved (AUC={best_auc:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"  -> Early stopping: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"  -> Early stopping triggered (Best AUC: {best_auc:.4f})")
            break

    logger.info(f"Fold {fold} done, Best AUC: {best_auc:.4f}")
    return best_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPH-Net single fold training')
    parser.add_argument('--train_csv', type=str, default='data/fold_0_train.csv')
    parser.add_argument('--val_csv', type=str, default='data/fold_0_val.csv')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--flow_loss_weight', type=float, default=0.1)
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'gated', 'simple'])
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--grad_accum_steps', type=int, default=1)

    args = parser.parse_args()

    train_fold(
        fold=args.fold,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root_dir=args.root_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        flow_loss_weight=args.flow_loss_weight,
        fusion_type=args.fusion_type,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps,
    )
