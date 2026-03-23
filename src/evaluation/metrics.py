import numpy as np
import cv2


class ComputeMetrics():

    def __init__(self):
        '''
        Initialise image quality metrics computation for evaluating watermark removal results

        Args:
            None

        Returns:
            None
        '''
        pass

    def psnr(self, clean_image, restored_image):
        '''
        Compute Peak Signal-to-Noise Ratio (PSNR) between a clean and a restored image.
        Higher PSNR indicates better reconstruction quality.

        Args:
            clean_image: numpy array of ground truth clean image (H x W x 3, uint8)
            restored_image: numpy array of model-restored image (H x W x 3, uint8)

        Returns:
            float representing PSNR in decibels (dB); returns 100.0 for identical images
        '''
        clean = clean_image.astype(np.float64)
        restored = restored_image.astype(np.float64)
        mse = np.mean((clean - restored) ** 2)
        # avoid log of zero for perfectly reconstructed images
        if mse == 0:
            return 100.0
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        return psnr

    def ssim(self, clean_image, restored_image):
        '''
        Compute the Structural Similarity Index (SSIM) between a clean and a restored image.
        SSIM captures luminance, contrast, and structural differences. Higher is better.

        Args:
            clean_image: numpy array of ground truth clean image (H x W x 3, uint8)
            restored_image: numpy array of model-restored image (H x W x 3, uint8)

        Returns:
            float representing mean SSIM in range [-1, 1]; 1.0 means identical images
        '''
        # numerical stability constants as per the original SSIM paper
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        clean_gray = cv2.cvtColor(clean_image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        restored_gray = cv2.cvtColor(restored_image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        # compute local means via Gaussian blur
        mu1 = cv2.GaussianBlur(clean_gray, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(restored_gray, (11, 11), 1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        # compute local variances and covariance
        sigma1_sq = cv2.GaussianBlur(clean_gray ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(restored_gray ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(clean_gray * restored_gray, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())

    def compute_all(self, clean_image, restored_image):
        '''
        Compute all image quality metrics for a single restored image

        Args:
            clean_image: numpy array of ground truth clean image (H x W x 3, uint8)
            restored_image: numpy array of model-restored image (H x W x 3, uint8)

        Returns:
            dictionary with keys 'psnr' and 'ssim' mapping to their respective float scores
        '''
        return {
            'psnr': self.psnr(clean_image, restored_image),
            'ssim': self.ssim(clean_image, restored_image)
        }
