import cv2
import numpy as np
import pickle


class TELEA():

    def __init__(self, radius=3):
        '''
        Setup TELEA model with radius

        Args:
            radius: radius of neighbourhood used to reconstruct each inpainted pixel

        Returns:
            None
        '''
        self._radius = radius
        self.name = 'TELEA'

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from image using TELEA algorithm

        Args:
            image: a numpy array which represents the watermarked image (H x W x 3, BGR)
            mask: a numpy array which represents the watermark mask (H x W)

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        # dilate mask to capture edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        restored = cv2.inpaint(image, dilated_mask, self._radius, cv2.INPAINT_TELEA)
        return restored

    def save(self, path):
        '''Save model configuration to disk'''
        with open(path, 'wb') as f:
            pickle.dump({'radius': self._radius}, f)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''Load model configuration from disk'''
        with open(path, 'rb') as f:
            config = pickle.load(f)
        return cls(**config)


class NSInpainting():

    def __init__(self, radius=3):
        '''
        Initialize OpenCV Navier-Stokes based inpainting model

        Args:
            radius: radius of the circular neighbourhood used to reconstruct each inpainted pixel

        Returns:
            None
        '''
        self._radius = radius
        self.name = 'NS Inpaint'

    def remove_watermark(self, image, mask):
        '''
        Remove watermark from image using the Navier-Stokes inpainting algorithm

        Args:
            image: a numpy array which represents the watermarked image (H x W x 3, BGR)
            mask: a numpy array which represents the watermark mask (H x W)

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        # dilate mask to get full coverage of watermark edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        restored = cv2.inpaint(image, dilated_mask, self._radius, cv2.INPAINT_NS)
        return restored

    def save(self, path):
        '''Save model to path
        
        Args:
            path: a directory to save model in

        Returns:
            NoneType
        '''
        with open(path, 'wb') as f:
            pickle.dump({'radius': self._radius}, f)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''Load model from path
        
        Args:
            path: a directory to load model from

        Returns:
            NoneType
        '''
        with open(path, 'rb') as f:
            config = pickle.load(f)
        return cls(**config)
