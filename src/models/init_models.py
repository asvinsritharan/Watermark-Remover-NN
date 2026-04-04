from src.models.classical_inpainting import TELEA, NSInpainting
from src.models.dictionary_learning_model import DictionaryLearningModel
from src.models.nmf_model import NMFModel
from src.models.cnn_autoencoder import CNNAutoencoder


class InitModels():

    def __init__(self):
        '''
        Setup all models for A/B comparison test. Set params for each model.

        Models used are: TELEA, Navier Strokes, Dictionary Learning, NMF, CNN U-NET

        Args:
            None

        Returns:
            None
        '''
        # create ordered dictionary of all models to be compared in the A/B experiment
        self.models = {
            'TELEA Inpainting': TELEA(radius=5),
            'NS Inpainting': NSInpainting(radius=5),
            'Dictionary Learning': DictionaryLearningModel(patch_size=8, n_components=64, alpha=1.0),
            'NMF': NMFModel(subsection=8, n_components=32),
            'CNN Autoencoder': CNNAutoencoder(n_epochs=50, lr=1e-3, batch_size=8),
        }
        print(f"Initialised {len(self.models)} models for A/B comparison: {list(self.models.keys())}")
