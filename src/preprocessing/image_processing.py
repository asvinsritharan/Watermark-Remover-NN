import cv2
import numpy as np
from pathlib import Path


class PrepareImageData():

    def __init__(self, clean_images_path, watermarked_images_path):
        '''
        Given clean_images_path and watermarked_images_path, load up the clean images from clean_images_path and watermarked_images_path and match images together based on name.
        File names must be identical for clean_images_path and watermarked_images_path, the extension can be different.
        
        The clean images are matched to files in the watermarked_image_path.

        image_0.jpg in clean_images_path is matches to image_0.jpeg in watermarked_images_path because the file names are the same.

        Args:
            clean_images_path: path to directory containing unwatermarked images
            watermarked_images_path: path to directory containing watermarked versions of the same images from clean_images_path

        Returns:
            None
        '''
        self._clean_images_path = Path(clean_images_path)
        self._watermarked_images_path = Path(watermarked_images_path)
        # load matched images with computed masks
        self.clean_images, self.watermarked_images, self.masks = self._load_paired_images()

    def _load_paired_images(self):
        '''
        Match clean:watermarked images based on filename. Create watermark mask by computing difference between
        clean and watermarked images.

        Args:
            None

        Returns:
            tuple of:
                - list of numpy arrays of clean images (H x W x 3, BGR)
                - list of numpy arrays of watermarked images (H x W x 3, BGR)
                - list of numpy arrays of binary watermark masks (H x W, uint8)
        '''
        clean_images = []
        watermarked_images = []
        masks = []
        # supported extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        clean_images_path = sorted([
            p for p in self._clean_images_path.iterdir()
            if p.suffix.lower() in extensions
        ])
        if not clean_images_path:
            print(f"No images found in clean images directory: {self._clean_images_path}")
            return clean_images, watermarked_images, masks
        # search for matching file names in watermarked folder
        watermarked_lookup = {
            p.stem.lower(): p
            for p in self._watermarked_images_path.iterdir()
            if p.suffix.lower() in extensions
        }
        matched = 0
        skipped = 0
        for clean_path in clean_images_path:
            # match by file name excluding extension
            watermarked_path = watermarked_lookup.get(clean_path.stem.lower())
            if watermarked_path is None:
                print(f"No matching watermarked image found for {clean_path.name}, skipping {clean_path.name}.")
                skipped += 1
                continue
            clean_image = cv2.imread(str(clean_path))
            watermarked_image = cv2.imread(str(watermarked_path))
            if clean_image is None or watermarked_image is None:
                print(f"Could not read image pair for {clean_path.name}, skipping.")
                skipped += 1
                continue
            # resize both images to a consistent resolution
            clean_image = cv2.resize(clean_image, (256, 256))
            watermarked_image = cv2.resize(watermarked_image, (256, 256))
            # derive the watermark mask from the pixel difference between the two images
            mask = self._compute_difference_mask(clean_image, watermarked_image)
            clean_images.append(clean_image)
            watermarked_images.append(watermarked_image)
            masks.append(mask)
            matched += 1
        print(f"Loaded {matched} matched image pairs. Skipped {skipped} unmatched files.")
        return clean_images, watermarked_images, masks

    def _compute_difference_mask(self, clean_image, watermarked_image):
        '''
        Compute a mask that is the difference between the clean and watermarked image

        Args:
            clean_image: numpy array of clean image (H x W x 3, BGR)
            watermarked_image: numpy array of watermarked image (H x W x 3, BGR)

        Returns:
            binary numpy array (H x W, uint8) where 255 = watermark region, 0 = background
        '''
        # compute absolute difference across all pixels
        diff = cv2.absdiff(clean_image, watermarked_image)
        # convert to greyscale
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # threshold: any pixel that changed by more than 10 IU is part of the watermark
        _, mask = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
        # dilate to capture watermark edges missed by the threshold
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # return mask
        return mask

    def _auto_detect_mask(self, image):
        '''
        Automatically detect the watermark region of an image using Otsu thresholding
        combined with low-saturation detection to catch both bright and semi-transparent watermarks.
        Defaults to fixed threshold if Otsu produces an improper result.

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)

        Returns:
            binary numpy array (H x W, uint8) where 255 = watermark region
        '''
        # greyscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect bright watermarks using a fixed threshold
        _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        # detect semi-transparent / grey watermarks
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, low_sat_mask = cv2.threshold(hsv[:, :, 1], 40, 255, cv2.THRESH_BINARY_INV)
        # combine: bright pixels that are also desaturated are strong watermark candidates
        combined = cv2.bitwise_and(bright_mask, low_sat_mask)
        mask_ratio = combined.sum() / 255.0 / (combined.shape[0] * combined.shape[1])
        # if combined mask is too sparse or too large, fall back to Otsu on grayscale
        if mask_ratio < 0.005 or mask_ratio > 0.5:
            _, combined = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(combined, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def apply_all_models(self, models, image_path, mask_path=None):
        '''
        Run model in models on watermarked image given in image_path and mask in mask_path.

        Args:
            models: dictionary mapping model name strings to model objects
            image_path: path of watermarked image to fix
            mask_path: optional path to a binary mask image

        Returns:
            dictionary mapping model name strings to watermark removed image numpy arrays (H x W x 3, BGR)
        '''
        # load image
        image = cv2.imread(str(Path(image_path).resolve()))
        # return error if the image couldn't be found
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        # resize image
        image = cv2.resize(image, (256, 256))
        # resize mask if it has been given. 
        if mask_path:
            mask = cv2.imread(str(Path(mask_path).resolve()), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = cv2.resize(mask, (256, 256))
        else:
            # detect mask if it wasn't given
            mask = self._auto_detect_mask(image)
        results = {}
        # apply model in models to the watermarked image
        for name, model in models.items():
            print(f"Applying {name}...")
            results[name] = model.remove_watermark(image.copy(), mask.copy())
        # return resulting images
        return results

    def apply_best_model(self, best_model, image_path, mask_path=None):
        '''
        Load a watermarked image, optionally load or auto-detect its mask, and apply the best model

        Args:
            best_model: trained model object
            image_path: path to the watermarked input image
            mask_path: optional path to a mask image

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        # Open image
        image = cv2.imread(str(Path(image_path).resolve()))
        # if image couldn't be opened then throw error
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        # reize image
        image = cv2.resize(image, (256, 256))
        # if mask is given, greyscale and resize
        if mask_path:
            mask = cv2.imread(str(Path(mask_path).resolve()), cv2.IMREAD_GRAYSCALE)
            # throw error if mask is not found
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = cv2.resize(mask, (256, 256))
        # if mask is not given then auto detect mask
        else:
            mask = self._auto_detect_mask(image)
        # remove watermark using model given
        return best_model.remove_watermark(image, mask)
