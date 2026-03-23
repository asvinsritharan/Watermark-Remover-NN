import cv2
import numpy as np
from pathlib import Path


class PrepareImageData():

    def __init__(self, clean_dir, watermarked_dir):
        '''
        Load matched pairs of clean and watermarked images from two directories.
        Files are matched by filename, so both directories must contain images with identical names.

        Args:
            clean_dir: path to directory containing clean (unwatermarked) images
            watermarked_dir: path to directory containing watermarked versions of the same images

        Returns:
            None
        '''
        self._clean_dir = Path(clean_dir)
        self._watermarked_dir = Path(watermarked_dir)
        # load matched image pairs and compute difference masks
        self.clean_images, self.watermarked_images, self.masks = self._load_paired_images()

    def _load_paired_images(self):
        '''
        Load all matched clean/watermarked image pairs by matching filenames across both directories.
        The watermark mask for each pair is derived by computing the absolute pixel difference
        between the clean and watermarked images and thresholding.

        Args:
            None

        Returns:
            tuple of:
                - clean_images: list of numpy arrays of clean images (H x W x 3, BGR)
                - watermarked_images: list of numpy arrays of watermarked images (H x W x 3, BGR)
                - masks: list of numpy arrays of binary watermark masks (H x W, uint8)
        '''
        clean_images = []
        watermarked_images = []
        masks = []
        # collect all image filenames present in the clean directory
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        clean_files = sorted([
            p for p in self._clean_dir.iterdir()
            if p.suffix.lower() in supported_exts
        ])
        if not clean_files:
            print(f"No images found in clean directory: {self._clean_dir}")
            return clean_images, watermarked_images, masks
        # build a stem -> path lookup for the watermarked directory to allow extension-agnostic matching
        watermarked_lookup = {
            p.stem.lower(): p
            for p in self._watermarked_dir.iterdir()
            if p.suffix.lower() in supported_exts
        }
        matched = 0
        skipped = 0
        for clean_path in clean_files:
            # match by stem only, ignoring file extension
            watermarked_path = watermarked_lookup.get(clean_path.stem.lower())
            if watermarked_path is None:
                print(f"No matching watermarked image found for {clean_path.name}, skipping.")
                skipped += 1
                continue
            clean_img = cv2.imread(str(clean_path))
            watermarked_img = cv2.imread(str(watermarked_path))
            if clean_img is None or watermarked_img is None:
                print(f"Could not read image pair for {clean_path.name}, skipping.")
                skipped += 1
                continue
            # resize both images to a consistent resolution
            clean_img = cv2.resize(clean_img, (256, 256))
            watermarked_img = cv2.resize(watermarked_img, (256, 256))
            # derive the watermark mask from the pixel difference between the two images
            mask = self._compute_difference_mask(clean_img, watermarked_img)
            clean_images.append(clean_img)
            watermarked_images.append(watermarked_img)
            masks.append(mask)
            matched += 1
        print(f"Loaded {matched} matched image pairs. Skipped {skipped} unmatched files.")
        return clean_images, watermarked_images, masks

    def _compute_difference_mask(self, clean_img, watermarked_img):
        '''
        Compute a binary mask indicating where the watermark is present by thresholding
        the absolute pixel difference between the clean and watermarked images

        Args:
            clean_img: numpy array of clean image (H x W x 3, BGR)
            watermarked_img: numpy array of watermarked image (H x W x 3, BGR)

        Returns:
            binary numpy array (H x W, uint8) where 255 = watermark region, 0 = background
        '''
        # compute per-pixel absolute difference across all channels
        diff = cv2.absdiff(clean_img, watermarked_img)
        # collapse to grayscale for thresholding
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # threshold: any pixel that changed by more than 10 intensity units is part of the watermark
        _, mask = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
        # dilate slightly to capture watermark edges missed by the threshold
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _auto_detect_mask(self, image):
        '''
        Automatically detect the watermark region of an image using Otsu thresholding
        combined with low-saturation detection to catch both bright and semi-transparent watermarks.
        Falls back to a fixed threshold if Otsu produces an implausible result.

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)

        Returns:
            binary numpy array (H x W, uint8) where 255 = watermark region
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect bright watermarks using a fixed threshold
        _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        # detect semi-transparent / grey watermarks via low HSV saturation
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
        Apply every model to a single watermarked image and return all outputs.
        The image and mask are loaded once and shared across all models.

        Args:
            models: dictionary mapping model name strings to model objects
            image_path: path to the watermarked input image
            mask_path: optional path to a binary mask image (white = watermark region)

        Returns:
            dictionary mapping model name strings to restored image numpy arrays (H x W x 3, BGR)
        '''
        image = cv2.imread(str(Path(image_path).resolve()))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.resize(image, (256, 256))
        if mask_path:
            mask = cv2.imread(str(Path(mask_path).resolve()), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = cv2.resize(mask, (256, 256))
        else:
            mask = self._auto_detect_mask(image)
        results = {}
        for name, model in models.items():
            print(f"Applying {name}...")
            results[name] = model.remove_watermark(image.copy(), mask.copy())
        return results

    def apply_best_model(self, best_model, image_path, mask_path=None):
        '''
        Load a watermarked image, optionally load or auto-detect its mask, and apply the best model

        Args:
            best_model: trained model object with a remove_watermark(image, mask) method
            image_path: path to the watermarked input image
            mask_path: optional path to a binary mask image (white = watermark region)

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        image = cv2.imread(str(Path(image_path).resolve()))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.resize(image, (256, 256))
        if mask_path:
            mask = cv2.imread(str(Path(mask_path).resolve()), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = cv2.resize(mask, (256, 256))
        else:
            mask = self._auto_detect_mask(image)
        return best_model.remove_watermark(image, mask)
