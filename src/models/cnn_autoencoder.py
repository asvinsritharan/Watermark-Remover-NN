import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class WatermarkDataset(Dataset):

    def __init__(self, watermarked_images, clean_images):
        '''
        PyTorch Dataset wrapping paired watermarked and clean image arrays

        Args:
            watermarked_images: list of numpy arrays of watermarked images (H x W x 3, BGR)
            clean_images: list of numpy arrays of clean images (H x W x 3, BGR)

        Returns:
            None
        '''
        self._watermarked = watermarked_images
        self._clean = clean_images

    def __len__(self):
        return len(self._watermarked)

    def __getitem__(self, idx):
        # normalise images to [0, 1] and convert from HWC to CHW format for PyTorch
        watermarked = self._watermarked[idx].astype(np.float32) / 255.0
        clean = self._clean[idx].astype(np.float32) / 255.0
        watermarked = torch.tensor(watermarked).permute(2, 0, 1)
        clean = torch.tensor(clean).permute(2, 0, 1)
        return watermarked, clean


class EncoderDecoderCNN(nn.Module):

    def __init__(self):
        '''
        Encoder-decoder CNN architecture for image-to-image watermark removal.
        The encoder progressively downsamples the input; the decoder upsamples back
        to the original resolution and outputs a clean image estimate.

        Args:
            None

        Returns:
            None
        '''
        super(EncoderDecoderCNN, self).__init__()
        # encoder: extract spatial features at decreasing resolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # decoder: reconstruct image at original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNAutoencoder():

    def __init__(self, n_epochs=10, lr=1e-3, batch_size=8):
        '''
        Initialize and configure a CNN autoencoder for supervised watermark removal

        Args:
            n_epochs: number of training epochs
            lr: learning rate for the Adam optimiser
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
        self._model = EncoderDecoderCNN().to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        self._criterion = nn.MSELoss()

    def fit(self, clean_images, watermarked_images, masks):
        '''
        Train the CNN autoencoder on paired watermarked and clean images using MSE loss

        Args:
            clean_images: list of numpy arrays of clean ground-truth images
            watermarked_images: list of numpy arrays of corresponding watermarked images
            masks: list of numpy arrays of watermark masks (unused; model learns implicitly)

        Returns:
            None
        '''
        dataset = WatermarkDataset(watermarked_images, clean_images)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        print(f"Training CNN Autoencoder on {len(dataset)} image pairs for {self._n_epochs} epochs...")
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
            print(f"  Epoch [{epoch + 1}/{self._n_epochs}], Loss: {avg_loss:.4f}")
        print("CNN Autoencoder training complete.")

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from an image using the trained CNN autoencoder

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
        Save the trained CNN weights and hyperparameters to disk using torch.save

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
