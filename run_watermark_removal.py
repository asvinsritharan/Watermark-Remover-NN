import argparse
import cv2
from pathlib import Path

from src.preprocessing.image_processing import PrepareImageData
from src.models.init_models import InitModels
from src.ab_testing.experiment import RunABTestingExperiment


class PerformWatermarkRemoval():

    def __init__(self, clean_dir, watermarked_dir, output_dir):
        '''
        Run full watermark removal pipeline: prepare data, train and compare models via A/B testing,
        then apply the best selected model for final watermark removal

        Args:
            clean_dir: path to directory containing clean (unwatermarked) images
            watermarked_dir: path to directory containing watermarked versions of the same images
            output_dir: path to directory where experiment results and final outputs will be saved

        Returns:
            None
        '''
        # load matched clean/watermarked image pairs from the two directories
        self._data_prep = PrepareImageData(clean_dir, watermarked_dir)
        # initialize all candidate model configurations
        self._model_configs = InitModels()
        # run A/B testing experiment across all models to determine the best performer
        self._experiment = RunABTestingExperiment(
            self._data_prep.clean_images,
            self._data_prep.watermarked_images,
            self._data_prep.masks,
            self._model_configs.models,
            output_dir
        )
        # store the best model selected by the experiment
        self.best_model_name = self._experiment.best_model_name
        self.best_model = self._experiment.best_model
        print(f"\nBest model selected for final application: {self.best_model_name}")

    def remove_watermark(self, image_path, mask_path=None, output_path=None):
        '''
        Apply the best model to remove the watermark from a given image

        Args:
            image_path: path to the watermarked input image
            mask_path: optional path to a binary mask indicating watermark region (white = watermark)
            output_path: optional path to save the restored image

        Returns:
            numpy array of the restored image
        '''
        restored = self._data_prep.apply_best_model(self.best_model, image_path, mask_path)
        # save restored image if output path is provided
        if output_path:
            cv2.imwrite(str(output_path), restored)
            print(f"Restored image saved to {output_path}")
        return restored

    def remove_watermark_all_models(self, image_path, mask_path=None, output_dir=None):
        '''
        Apply every trained model to a single watermarked image and save each output.
        Outputs are saved as <model_name>_output.png inside a per_model_outputs subdirectory.

        Args:
            image_path: path to the watermarked input image
            mask_path: optional path to a binary mask indicating watermark region (white = watermark)
            output_dir: directory under which per_model_outputs/ will be created

        Returns:
            dictionary mapping model name strings to restored image numpy arrays
        '''
        all_models = self._experiment._models
        results = self._data_prep.apply_all_models(all_models, image_path, mask_path)
        if output_dir:
            per_model_dir = Path(output_dir) / 'per_model_outputs'
            per_model_dir.mkdir(parents=True, exist_ok=True)
            for name, restored in results.items():
                filename = name.lower().replace(' ', '_') + '_output.png'
                save_path = per_model_dir / filename
                cv2.imwrite(str(save_path), restored)
                print(f"Saved {name} output to {save_path}")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watermark Removal using Machine Learning')
    parser.add_argument('--clean_dir', type=str, default='data/clean_images',
                        help='Directory of clean (unwatermarked) images')
    parser.add_argument('--watermarked_dir', type=str, default='data/watermarked_images',
                        help='Directory of watermarked images (filenames must match clean_dir)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for experiment results and plots')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to a watermarked image to process with the best model')
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to binary watermark mask for the test image')
    args = parser.parse_args()

    pipeline = PerformWatermarkRemoval(args.clean_dir, args.watermarked_dir, args.output_dir)

    if args.test_image:
        output_path = Path(args.output_dir) / 'final_output.png'
        pipeline.remove_watermark(args.test_image, args.mask, output_path)
        pipeline.remove_watermark_all_models(args.test_image, args.mask, args.output_dir)
