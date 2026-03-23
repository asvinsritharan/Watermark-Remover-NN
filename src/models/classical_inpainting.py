import cv2
import numpy as np
import pickle
from pathlib import Path


class TELEAInpainting():

    def __init__(self, inpaint_radius=3):
        '''
        Initialize OpenCV TELEA (Fast Marching Method) inpainting model

        Args:
            inpaint_radius: radius of the circular neighbourhood used to reconstruct each inpainted pixel

        Returns:
            None
        '''
        self._inpaint_radius = inpaint_radius
        self.name = 'TELEA Inpainting'

    def fit(self, clean_images, watermarked_images, masks):
        '''
        TELEA inpainting is a non-parametric algorithm and requires no training

        Args:
            clean_images: list of numpy arrays of clean images (unused)
            watermarked_images: list of numpy arrays of watermarked images (unused)
            masks: list of numpy arrays of watermark masks (unused)

        Returns:
            None
        '''
        # no training required for classical inpainting
        pass

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from image using the TELEA inpainting algorithm

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)
            mask: numpy array of binary watermark mask (H x W), 255 = watermark region

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        # slightly dilate the mask to ensure full coverage of watermark edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        restored = cv2.inpaint(image, dilated_mask, self._inpaint_radius, cv2.INPAINT_TELEA)
        return restored

    def save(self, path):
        '''Save model configuration to disk'''
        with open(path, 'wb') as f:
            pickle.dump({'inpaint_radius': self._inpaint_radius}, f)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''Load model configuration from disk'''
        with open(path, 'rb') as f:
            config = pickle.load(f)
        return cls(**config)


class NSInpainting():

    def __init__(self, inpaint_radius=3):
        '''
        Initialize OpenCV Navier-Stokes based inpainting model

        Args:
            inpaint_radius: radius of the circular neighbourhood used to reconstruct each inpainted pixel

        Returns:
            None
        '''
        self._inpaint_radius = inpaint_radius
        self.name = 'NS Inpainting'

    def fit(self, clean_images, watermarked_images, masks):
        '''
        Navier-Stokes inpainting is a non-parametric algorithm and requires no training

        Args:
            clean_images: list of numpy arrays of clean images (unused)
            watermarked_images: list of numpy arrays of watermarked images (unused)
            masks: list of numpy arrays of watermark masks (unused)

        Returns:
            None
        '''
        # no training required for classical inpainting
        pass

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from image using the Navier-Stokes inpainting algorithm

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)
            mask: numpy array of binary watermark mask (H x W), 255 = watermark region

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        # slightly dilate the mask to ensure full coverage of watermark edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        restored = cv2.inpaint(image, dilated_mask, self._inpaint_radius, cv2.INPAINT_NS)
        return restored

    def save(self, path):
        '''Save model configuration to disk'''
        with open(path, 'wb') as f:
            pickle.dump({'inpaint_radius': self._inpaint_radius}, f)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''Load model configuration from disk'''
        with open(path, 'rb') as f:
            config = pickle.load(f)
        return cls(**config)
