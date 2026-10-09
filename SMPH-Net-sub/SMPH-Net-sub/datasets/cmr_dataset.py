import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import nibabel as nib


class CMRDataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, num_frames=16):
        """
        Args:
            csv_file: CSV file path containing data paths and labels.
            root_dir: Data root directory.
            num_frames: Number of frames to sample per sequence.
        """
        self.num_frames = num_frames

        if csv_file is not None:
            self.df = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.use_csv = True
        else:
            self.root_dir = root_dir
            self.samples = os.listdir(root_dir)
            self.use_csv = False

    def load_nifti(self, path):
        """Load NIfTI file and extract frames."""
        nii = nib.load(path)
        data = nii.get_fdata()

        if len(data.shape) == 4:
            # 4D data (H, W, D, T) - 3D volume time series
            frames = data.shape[-1]
            idxs = np.linspace(0, frames-1, min(self.num_frames, frames)).astype(int)
            selected_frames = data[:, :, :, idxs]

            frames_list = []
            for i in range(selected_frames.shape[-1]):
                mid_slice = selected_frames[:, :, selected_frames.shape[2]//2, i]
                mid_slice = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)
                mid_slice = cv2.resize(mid_slice, (224, 224))
                frames_list.append(mid_slice)
        elif len(data.shape) == 3:
            # 3D data (H, W, T) - 2D image time series
            frames = data.shape[-1]
            idxs = np.linspace(0, frames-1, min(self.num_frames, frames)).astype(int)
            selected_frames = data[:, :, idxs]

            frames_list = []
            for i in range(selected_frames.shape[-1]):
                frame = selected_frames[:, :, i]
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                frame = cv2.resize(frame, (224, 224))
                frames_list.append(frame)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        result = np.stack(frames_list, axis=0)
        return result

    def load_sequence(self, path):
        """Load and preprocess image sequence."""
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            return self.load_nifti(path)
        else:
            imgs = sorted(os.listdir(path))
            idxs = np.linspace(0, len(imgs)-1, self.num_frames).astype(int)
            frames = []
            for i in idxs:
                img = cv2.imread(os.path.join(path, imgs[i]), 0)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                frames.append(img)
            return np.stack(frames)

    def __len__(self):
        if self.use_csv:
            return len(self.df)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_csv:
            row = self.df.iloc[idx]
            sample_path = row['path']
            label = row['label']
            path = os.path.join(self.root_dir, sample_path) if self.root_dir else sample_path
        else:
            sample = self.samples[idx]
            path = os.path.join(self.root_dir, sample)
            label = 1 if "PH" in sample else 0

        x = self.load_sequence(path)
        x = torch.tensor(x).unsqueeze(1).float()
        y = torch.tensor(label).float()
        return x, y

    def collate_fn(self, batch):
        """Custom collate function to stack batch as (B, T, C, H, W)."""
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels
