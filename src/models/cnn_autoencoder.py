import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class WatermarkDataset(Dataset):

    def __init__(self, clean_images, watermarked_images, augment=False):
        '''
        Given clean_images and watermarked_images, create a PyTorch dataset pairing the watermarked image and clean immage arrays together
        Given augment, apply random flips during training if True.

        Args:
            watermarked_images: list of numpy arrays of watermarked images (H x W x 3, BGR)
            clean_images: list of numpy arrays of clean images (H x W x 3, BGR)
            augment: boolean to apply random flips during training

        Returns:
            None
        '''
        self._clean = clean_images
        self._watermarked = watermarked_images
        self._augment = augment

    def __len__(self):
        return len(self._watermarked)

    def __getitem__(self, idx):
        # normalise images to [0, 1] and convert from HWC to CHW format for PyTorch
        watermarked = torch.tensor(self._watermarked[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        clean = torch.tensor(self._clean[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        if self._augment:
            # apply the same random flips to both images
            if random.random() > 0.5:
                watermarked = TF.hflip(watermarked)
                clean = TF.hflip(clean)
            if random.random() > 0.5:
                watermarked = TF.vflip(watermarked)
                clean = TF.vflip(clean)
        return watermarked, clean


def _conv_block(input_channel_count, output_channel_count):
    '''
    Given input and output channels, create a U-Net double convolution block.

    Args:
        input_channel_count: number of input channels
        output_channel_count: number of output channels

    Returns:
        nn.Sequential U-Net double convolution block
    '''
    return nn.Sequential(
        nn.Conv2d(input_channel_count, output_channel_count, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channel_count),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channel_count, output_channel_count, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(output_channel_count),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):

    def __init__(self):
        '''
        Create U-Net encoder/decoder for image watermark removal.

        Args:
            None

        Returns:
            None
        '''
        super(UNet, self).__init__()
        # encoder path: progressively downsample and increase channel depth
        self.encoder_1 = _conv_block(3, 64)
        self.encoder_2 = _conv_block(64, 128)
        self.encoder_3 = _conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        # bottleneck: deepest feature representation
        self.bottleneck = _conv_block(256, 512)
        # decoder path: upsample and concatenate with matching encoder skip connections
        self.upsampler_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_3 = _conv_block(512, 256)
        self.upsampler_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_2 = _conv_block(256, 128)
        self.upsampler_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_1 = _conv_block(128, 64)
        # output head: map 64 channels back to 3 RGB channels
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Create encoder/decoder for image watermark removal.

        Args:
            None
        Returns:
            nn.Sequential U-Net Convolution block
        '''
        # encoder: save each level's output for skip connections
        encoder_1 = self.encoder_1(x)
        encoder_2 = self.encoder_2(self.pool(encoder_1))
        encoder_3 = self.encoder_3(self.pool(encoder_2))
        # bottleneck
        bottleneck = self.bottleneck(self.pool(encoder_3))
        # decoder: upsample then connect matching encoder output
        decoder_3 = self.decoder_3(torch.cat([self.upsampler_3(bottleneck), encoder_3], dim=1))
        decoder_2 = self.decoder_2(torch.cat([self.upsampler_2(decoder_3), encoder_2], dim=1))
        decoder_1 = self.decoder_1(torch.cat([self.upsampler_1(decoder_2), encoder_1], dim=1))
        return self.output_conv(decoder_1)


class CNNAutoencoder():

    def __init__(self, n_epochs=50, lr=0.001, batch_size=8):
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
        Train the U-Net on paired watermarked and clean images using L1 loss function.

        Args:
            clean_images: list of numpy arrays of clean images
            watermarked_images: list of numpy arrays of corresponding watermarked images
            masks: _placeholder_variable

        Returns:
            None
        '''
        # create dataset
        dataset = WatermarkDataset(watermarked_images, clean_images, augment=True)
        # load datset
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        print(f"Training CNN Autoencoder (U-Net) on {len(dataset)} image pairs for {self._n_epochs} epochs...")
        # run training of CNN
        self._model.train()
        for epoch in range(self._n_epochs):
            epoch_loss = 0.0
            # for each watermarked/clean image batch
            for watermarked_batch, clean_batch in dataloader:
                # load to gpu
                watermarked_batch = watermarked_batch.to(self._device)
                clean_batch = clean_batch.to(self._device)
                # predict restored image from watermarked input -> forward
                self._optimizer.zero_grad()
                restored_batch = self._model(watermarked_batch)
                # calculate loss for current batch after forward pass
                loss = self._criterion(restored_batch, clean_batch)
                # update model weights ._ backwards
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            # calculate loss over all batches
            avg_loss = epoch_loss / len(dataloader)
            # set step size
            self._scheduler.step(avg_loss)
            # get current learning rate
            current_lr = self._optimizer.param_groups[0]['lr']
            # print epoch, avg loss and current learning rate for epoch
            print(f"  Epoch [{epoch + 1}/{self._n_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        # print completion message
        print("CNN Autoencoder training complete.")

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from given image using CNN trained model

        Args:
            image: a numpy array which represents the watermarked image (H x W x 3, BGR)
            mask: a numpy array which represents the watermark mask (H x W)

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
        # return image without watermark
        return restored

    def save(self, path):
        '''
        Save the trained U-Net weights and hyperparameters to path using torch.save
        
        Args:
            path: a directory to save model in

        Returns:
            NoneType
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
        Load a trained CNN Autoencoder from path
 
        Args:
            path: a directory to load model from

        Returns:
            Trained CNNAutoencoder model
        '''
        data = torch.load(path, map_location='cpu')
        instance = cls(n_epochs=data['n_epochs'], lr=data['lr'], batch_size=data['batch_size'])
        instance._model.load_state_dict(data['model_state_dict'])
        return instance
