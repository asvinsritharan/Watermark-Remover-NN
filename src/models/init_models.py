from src.models.classical_inpainting import TELEAInpainting, NSInpainting
from src.models.dictionary_learning_model import DictionaryLearningModel
from src.models.nmf_model import NMFModel
from src.models.cnn_autoencoder import CNNAutoencoder


class InitModels():

    def __init__(self):
        '''
        Initialise all candidate watermark removal models for A/B comparison.
        Each model exposes a consistent fit(clean, watermarked, masks) and
        remove_watermark(image, mask) interface.

        Models included:
            - TELEA Inpainting (classical, non-parametric)
            - NS Inpainting (classical, non-parametric)
            - Dictionary Learning (patch-based sparse coding)
            - NMF (patch-based non-negative matrix factorisation)
            - CNN Autoencoder (deep learning, supervised)

        Args:
            None

        Returns:
            None
        '''
        # create ordered dictionary of all models to be compared in the A/B experiment
        self.models = {
            'TELEA Inpainting': TELEAInpainting(inpaint_radius=5),
            'NS Inpainting': NSInpainting(inpaint_radius=5),
            'Dictionary Learning': DictionaryLearningModel(patch_size=8, n_components=64, alpha=1.0),
            'NMF': NMFModel(patch_size=8, n_components=32),
            'CNN Autoencoder': CNNAutoencoder(n_epochs=50, lr=1e-3, batch_size=8),
        }
        print(f"Initialised {len(self.models)} models for A/B comparison: {list(self.models.keys())}")
