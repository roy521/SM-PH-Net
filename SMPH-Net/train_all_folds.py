"""SMPH-Net 5-fold cross-validation training script.

Usage:
    python train_all_folds.py --data_dir data --num_epochs 50 --batch_size 4
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime


def train_all_folds(data_dir='data', num_folds=5, num_epochs=50, batch_size=4,
                    lr=1e-4, num_frames=16, output_dir='outputs',
                    flow_loss_weight=0.1, fusion_type='cross_attention',
                    early_stopping_patience=15):
    """Train all folds sequentially."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f'all_folds_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"Training {num_folds} folds ...")
    print(f"Data dir: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR: {lr}")
    print(f"Frames: {num_frames}")
    print(f"Output: {run_dir}")

    for fold in range(num_folds):
        train_csv = os.path.join('data', f'fold_{fold}_train.csv')
        val_csv = os.path.join('data', f'fold_{fold}_val.csv')

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            print(f"\nError: Fold {fold} CSV files not found!")
            print("Run data_split.py first to generate data splits.")
            return

        fold_output_dir = os.path.join(run_dir, f'fold_{fold}')
        os.makedirs(fold_output_dir, exist_ok=True)

        cmd = [
            sys.executable, 'train.py',
            '--train_csv', train_csv,
            '--val_csv', val_csv,
            '--root_dir', data_dir,
            '--num_epochs', str(num_epochs),
            '--batch_size', str(batch_size),
            '--lr', str(lr),
            '--fold', str(fold),
            '--num_frames', str(num_frames),
            '--output_dir', fold_output_dir,
            '--flow_loss_weight', str(flow_loss_weight),
            '--fusion_type', fusion_type,
            '--early_stopping_patience', str(early_stopping_patience),
        ]

        print(f"\n{'='*60}")
        print(f"Training Fold {fold}/{num_folds - 1}")
        print(f"{'='*60}")

        subprocess.run(cmd, check=True)
        print(f"Fold {fold} done")

    print(f"\n{'='*60}")
    print(f"All {num_folds} folds completed!")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPH-Net 5-fold training')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--flow_loss_weight', type=float, default=0.1)
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'gated', 'simple'])
    parser.add_argument('--early_stopping_patience', type=int, default=15)

    args = parser.parse_args()

    train_all_folds(
        data_dir=args.data_dir,
        num_folds=args.num_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        flow_loss_weight=args.flow_loss_weight,
        fusion_type=args.fusion_type,
        early_stopping_patience=args.early_stopping_patience,
    )
