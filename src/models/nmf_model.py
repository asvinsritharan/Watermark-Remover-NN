import numpy as np
import joblib
from sklearn.decomposition import NMF


class NMFModel():

    def __init__(self, subsection=8, n_components=32):
        '''
        Initialize a patch-based Non-negative Matrix Factorisation (NMF) model for watermark removal.
        Use clean image patches with different colour channels for NMF component learning. Watermarked patches
        are then reconstructed by projecting onto the learned component space.

        Args:
            subsection: dimension oheight and width of square image patches to extract
            n_components: number of NMF components to learn per channel

        Returns:
            None
        '''
        self._subsection = subsection
        self._n_components = n_components
        self.name = 'NMF'
        # initialise one NMF model per colour channel
        self._nmf_models = []
        for _ in range(3):
            self._nmf_models.append(NMF(n_components=n_components, max_iter=350, random_state=64))
        self._components = None

    def _extract_channel_patches(self, image, channel, mask=None):
        '''
        Extract flattened patches from a single colour channel of an image

        Args:
            image: numpy array of image (H x W x 3, BGR)
            channel: integer index of colour channel to extract (0=B, 1=G, 2=R)
            mask: optional binary numpy array (H x W); patches where mask mean > 50 are skipped

        Returns:
            numpy array of shape (n_patches, subsection^2) with float32 values in [0, 1]
        '''
        h, w = image.shape[:2]
        p = self._subsection
        patches = []
        for i in range(0, h - p, p):
            for j in range(0, w - p, p):
                # skip patches that fall inside the watermark region
                if mask is not None:
                    if mask[i:i + p, j:j + p].mean() > 50:
                        continue
                patch = image[i:i + p, j:j + p, channel].astype(np.float32) / 255.0
                patches.append(patch.flatten())
        if not patches:
            return np.empty((0, self._subsection * self._subsection), dtype=np.float32)
        return np.array(patches)

    def fit(self, clean_images, watermarked_images, masks):
        '''
        Learn NMF components from clean image patches for each colour channel independently

        Args:
            clean_images: list of numpy arrays of clean images
            watermarked_images: list of numpy arrays of watermarked images (unused)
            masks: list of numpy arrays of binary watermark masks

        Returns:
            None
        '''
        self._components = []
        for c in range(3):
            channel_patches = []
            for clean_img, mask in zip(clean_images, masks):
                patches = self._extract_channel_patches(clean_img, c, mask=mask)
                channel_patches.append(patches)
            channel_patches = [p for p in channel_patches if len(p) > 0]
            if not channel_patches:
                raise ValueError(f"No clean patches found for channel {c} — watermark masks may cover entire images.")
            all_patches = np.vstack(channel_patches)
            # subsample to a manageable number of patches for efficiency
            if len(all_patches) > 5000:
                idx = np.random.choice(len(all_patches), 5000, replace=False)
                all_patches = all_patches[idx]
            print(f"Fitting NMF on channel {c} with {len(all_patches)} patches...")
            self._nmf_models[c].fit(all_patches)
            self._components.append(self._nmf_models[c].components_)
        print("NMF fitting complete.")

    def remove_watermark(self, image, mask):
        '''
        Reconstruct watermarked sections channel-by-channel using the learned NMF components

        Args:
            image: numpy array of watermarked image (H x W x 3, BGR)
            mask: numpy array of binary watermark mask (H x W), 255 = watermark region

        Returns:
            numpy array of the restored image (H x W x 3, BGR)
        '''

        height, width = image.shape[:2]
        subsection = self._subsection
        restored = image.copy().astype(np.float32) / 255.0
        for i in range(0, height - subsection, subsection):
            for j in range(0, width - subsection, subsection):
                patch_mask = mask[i:i + subsection, j:j + subsection]
                # only reconstruct patches that overlap with the watermark
                if patch_mask.mean() > 30:
                    for c in range(3):
                        patch = image[i:i + subsection, j:j + subsection, c].astype(np.float32) / 255.0
                        patch_flat = patch.flatten().reshape(1, -1)
                        # project to NMF space and reconstruct
                        H = self._nmf_models[c].transform(patch_flat)
                        reconstructed = (H @ self._components[c]).flatten()
                        reconstructed = np.clip(reconstructed, 0, 1)
                        restored[i:i + subsection, j:j + subsection, c] = reconstructed.reshape(subsection, subsection)
        return (restored * 255).astype(np.uint8)

    def save(self, path):
        '''
        Save the trained NMF models and components to path using joblib

        Args:
            path: file path to save the model to (e.g. results/saved_models/nmf.pkl)

        Returns:
            None
        '''
        joblib.dump({'nmf_models': self._nmf_models, 'components': self._components,
                     'subsection': self._subsection, 'n_components': self._n_components}, path)
        print(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path):
        '''
        Load a trained NMF model from path

        Args:
            path: file path to load the model from

        Returns:
            fitted NMF model
        '''
        data = joblib.load(path)
        instance = cls(subsection=data['subsection'], n_components=data['n_components'])
        instance._nmf_models = data['nmf_models']
        instance._components = data['components']
        return instance
