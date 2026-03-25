import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class WatermarkDataset(Dataset):

    def __init__(self, watermarked_images, clean_images, augment=False):
        '''
        PyTorch Dataset wrapping paired watermarked and clean image arrays,
        with optional random horizontal/vertical flip augmentation applied
        consistently to both images in each pair.

        Args:
            watermarked_images: list of numpy arrays of watermarked images (H x W x 3, BGR)
            clean_images: list of numpy arrays of clean images (H x W x 3, BGR)
            augment: whether to apply random flips during training

        Returns:
            None
        '''
        self._watermarked = watermarked_images
        self._clean = clean_images
        self._augment = augment

    def __len__(self):
        return len(self._watermarked)

    def __getitem__(self, idx):
        # normalise images to [0, 1] and convert from HWC to CHW format for PyTorch
        watermarked = torch.tensor(self._watermarked[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        clean = torch.tensor(self._clean[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        if self._augment:
            # apply the same random flips to both images to preserve alignment
            if random.random() > 0.5:
                watermarked = TF.hflip(watermarked)
                clean = TF.hflip(clean)
            if random.random() > 0.5:
                watermarked = TF.vflip(watermarked)
                clean = TF.vflip(clean)
        return watermarked, clean


def _conv_block(in_ch, out_ch):
    '''
    Standard U-Net double convolution block: Conv → BN → ReLU → Conv → BN → ReLU

    Args:
        in_ch: number of input channels
        out_ch: number of output channels

    Returns:
        nn.Sequential block
    '''
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):

    def __init__(self):
        '''
        U-Net encoder-decoder architecture with skip connections for image-to-image watermark removal.
        Skip connections concatenate encoder feature maps directly into the decoder at each resolution,
        preserving spatial detail that would otherwise be lost through downsampling.

        Args:
            None

        Returns:
            None
        '''
        super(UNet, self).__init__()
        # encoder path: progressively downsample and increase channel depth
        self.enc1 = _conv_block(3, 64)
        self.enc2 = _conv_block(64, 128)
        self.enc3 = _conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        # bottleneck: deepest feature representation
        self.bottleneck = _conv_block(256, 512)
        # decoder path: upsample and concatenate with matching encoder skip connections
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _conv_block(128, 64)
        # output head: map 64 channels back to 3 RGB channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoder: save each level's output for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # bottleneck
        b = self.bottleneck(self.pool(e3))
        # decoder: upsample then concatenate matching encoder output
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output_conv(d1)


class CNNAutoencoder():

    def __init__(self, n_epochs=50, lr=1e-3, batch_size=8):
        '''
        Initialize and configure a U-Net based CNN for supervised watermark removal.
        Uses L1 loss for sharp reconstructions and a ReduceLROnPlateau scheduler
        to lower the learning rate when training plateaus.

        Args:
            n_epochs: number of training epochs
            lr: initial learning rate for the Adam optimiser
            batch_size: number of image pairs per training batch

        Returns:
            None
        '''
        self._n_epochs = n_epochs
        self._lr = lr
        self._batch_size = batch_size
        self.name = 'CNN Autoencoder'
        # use GPU if available, otherwise fall back to CPU
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = UNet().to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        # L1 loss produces sharper outputs than MSE by not penalising large errors quadratically
        self._criterion = nn.L1Loss()
        # halve LR when training loss stops improving for 5 consecutive epochs
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode='min', factor=0.5, patience=5
        )

    def fit(self, clean_images, watermarked_images, masks):
        '''
        Train the U-Net on paired watermarked and clean images using L1 loss.
        Augmentation is applied during training to improve generalisation on small datasets.

        Args:
            clean_images: list of numpy arrays of clean ground-truth images
            watermarked_images: list of numpy arrays of corresponding watermarked images
            masks: list of numpy arrays of watermark masks (unused; model learns implicitly)

        Returns:
            None
        '''
        dataset = WatermarkDataset(watermarked_images, clean_images, augment=True)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        print(f"Training CNN Autoencoder (U-Net) on {len(dataset)} image pairs for {self._n_epochs} epochs...")
        self._model.train()
        for epoch in range(self._n_epochs):
            epoch_loss = 0.0
            for watermarked_batch, clean_batch in dataloader:
                watermarked_batch = watermarked_batch.to(self._device)
                clean_batch = clean_batch.to(self._device)
                # forward pass: predict restored image from watermarked input
                self._optimizer.zero_grad()
                restored_batch = self._model(watermarked_batch)
                loss = self._criterion(restored_batch, clean_batch)
                # backward pass: update model weights
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            self._scheduler.step(avg_loss)
            current_lr = self._optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch + 1}/{self._n_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        print("CNN Autoencoder training complete.")

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from an image using the trained U-Net

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)
            mask: numpy array of binary watermark mask (H x W), 255 = watermark region (unused at inference)

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        self._model.eval()
        with torch.no_grad():
            # normalise, convert to CHW tensor, and add batch dimension
            img_tensor = torch.tensor(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self._device)
            restored_tensor = self._model(img_tensor)
            # convert output back to HWC numpy array in [0, 255]
            restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = (restored * 255).astype(np.uint8)
        return restored

    def save(self, path):
        '''
        Save the trained U-Net weights and hyperparameters to disk using torch.save

        Args:
            path: file path to save the model to (e.g. results/saved_models/cnn_autoencoder.pt)

        Returns:
            None
        '''
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'n_epochs': self._n_epochs,
            'lr': self._lr,
            'batch_size': self._batch_size,
        }, path)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''
        Load a trained CNN Autoencoder from disk

        Args:
            path: file path to load the model from

        Returns:
            CNNAutoencoder instance with restored weights
        '''
        data = torch.load(path, map_location='cpu')
        instance = cls(n_epochs=data['n_epochs'], lr=data['lr'], batch_size=data['batch_size'])
        instance._model.load_state_dict(data['model_state_dict'])
        return instance
