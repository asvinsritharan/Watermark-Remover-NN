import numpy as np
import joblib
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit


class DictionaryLearningModel():

    def __init__(self, patch_size=8, n_components=64, alpha=1.0):
        '''
        Initialize a patch-based dictionary learning model for watermark removal.
        A dictionary of image patches is learned from clean regions; watermarked patches
        are then reconstructed via sparse coding over this learned dictionary.

        Args:
            patch_size: height and width of square image patches to extract
            n_components: number of dictionary atoms (basis patches) to learn
            alpha: sparsity regularisation parameter for dictionary learning

        Returns:
            None
        '''
        self._patch_size = patch_size
        self._n_components = n_components
        self._alpha = alpha
        self.name = 'Dictionary Learning'
        # initialize sklearn DictionaryLearning estimator
        self._dict_learner = DictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            max_iter=200,
            random_state=42,
            n_jobs=-1
        )
        self._dictionary = None

    def _extract_patches(self, image, mask=None):
        '''
        Extract flattened non-overlapping patches from an image, skipping patches that
        overlap significantly with a watermark mask if one is provided

        Args:
            image: numpy array of image (H x W x 3, BGR)
            mask: optional binary numpy array (H x W); patches where mask mean > 50 are skipped

        Returns:
            numpy array of shape (n_patches, patch_size^2 * 3) with float32 values in [0, 1]
        '''
        height, width = image.shape[:2]
        patch_size = self._patch_size
        patches = []
        for x in range(0, height - patch_size, patch_size):
            for y in range(0, width - patch_size, patch_size):
                # skip patches that contain watermark
                if mask is not None:
                    if mask[x:x + patch_size, y:y + patch_size].mean() > 50:
                        continue
                patch = image[x:x + patch_size, y:y + patch_size].astype(np.float32) / 255.0
                patches.append(patch.flatten())
        if not patches:
            return np.empty((0, self._patch_size * self._patch_size * 3), dtype=np.float32)
        return np.array(patches)

    def fit(self, clean_images, watermarked_images, masks):
        '''
        Learn a patch dictionary from clean image patches, excluding watermark regions

        Args:
            clean_images: list of numpy arrays of clean images
            watermarked_images: list of numpy arrays of watermarked images (unused)
            masks: list of numpy arrays of binary watermark masks

        Returns:
            None
        '''
        # collect patches from all clean images, excluding watermark regions
        all_patches = []
        for clean_img, mask in zip(clean_images, masks):
            patches = self._extract_patches(clean_img, mask=mask)
            all_patches.append(patches)
        all_patches = [p for p in all_patches if len(p) > 0]
        if not all_patches:
            raise ValueError("No clean patches found for training — watermark masks may cover entire images.")
        all_patches = np.vstack(all_patches)
        # subsample to a manageable number of patches for efficiency
        if len(all_patches) > 5000:
            idx = np.random.choice(len(all_patches), 5000, replace=False)
            all_patches = all_patches[idx]
        print(f"Fitting Dictionary Learning on {len(all_patches)} patches...")
        self._dict_learner.fit(all_patches)
        self._dictionary = self._dict_learner.components_
        print("Dictionary Learning fitting complete.")

    def remove_watermark(self, image, mask):
        '''
        Remove watermark and reconstruct watermark subgrid using learned dictionary

        Args:
            image: a numpy array which represents the watermarked image (H x W x 3, BGR)
            mask: a numpy array which represents the watermark mask (H x W)

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''
        h, w = image.shape[:2]
        p = self._patch_size
        restored = image.astype(np.float32) / 255.0
        # use Orthogonal Matching Pursuit for sparse patch reconstruction
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
        for i in range(0, h - p, p):
            for j in range(0, w - p, p):
                patch_mask = mask[i:i + p, j:j + p]
                # only reconstruct patches that overlap with the watermark
                if patch_mask.mean() > 30:
                    patch = image[i:i + p, j:j + p].astype(np.float32) / 255.0
                    patch_flat = patch.flatten().reshape(1, -1)
                    # encode patch as a sparse combination of dictionary atoms
                    omp.fit(self._dictionary.T, patch_flat.T)
                    coeffs = omp.coef_
                    reconstructed = (self._dictionary.T @ coeffs).flatten()
                    reconstructed = np.clip(reconstructed, 0, 1)
                    restored[i:i + p, j:j + p] = reconstructed.reshape(p, p, 3)
        return (restored * 255).astype(np.uint8)

    def save(self, path):
        '''
        Save the trained dictionary to disk using joblib

        Args:
            path: file path to save the model to (e.g. results/saved_models/dict_learning.pkl)

        Returns:
            None
        '''
        joblib.dump({'dict_learner': self._dict_learner, 'dictionary': self._dictionary,
                     'patch_size': self._patch_size, 'n_components': self._n_components,
                     'alpha': self._alpha}, path)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''
        Load a trained Dictionary Learning model from disk

        Args:
            path: file path to load the model from

        Returns:
            DictionaryLearningModel instance with restored state
        '''
        data = joblib.load(path)
        instance = cls(patch_size=data['patch_size'], n_components=data['n_components'], alpha=data['alpha'])
        instance._dict_learner = data['dict_learner']
        instance._dictionary = data['dictionary']
        return instance
