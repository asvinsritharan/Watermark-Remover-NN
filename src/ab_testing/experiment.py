import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

from src.evaluation.metrics import ComputeMetrics


class RunABTestingExperiment():

    def __init__(self, clean_images, watermarked_images, masks, models, output_dir):
        '''
        Run an A/B testing experiment comparing all watermark removal models on a held-out test set.
        Models are trained on an 80% split, evaluated on the remaining 20%, then ranked using
        ANOVA and pairwise Welch's t-tests. The highest mean PSNR model is selected.

        Args:
            clean_images: list of numpy arrays of ground truth clean images
            watermarked_images: list of numpy arrays of corresponding watermarked images
            masks: list of numpy arrays of binary watermark masks
            models: dictionary mapping model name strings to model objects
            output_dir: path to directory for saving result reports and comparison plots

        Returns:
            None
        '''
        self._clean_images = clean_images
        self._watermarked_images = watermarked_images
        self._masks = masks
        self._models = models
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_computer = ComputeMetrics()
        # split into 80% train and 20% test
        n = len(clean_images)
        split = int(0.8 * n)
        self._train_clean = clean_images[:split]
        self._train_watermarked = watermarked_images[:split]
        self._train_masks = masks[:split]
        self._test_clean = clean_images[split:]
        self._test_watermarked = watermarked_images[split:]
        self._test_masks = masks[split:]
        self._saved_models_dir = self._output_dir / 'saved_models'
        self._saved_models_dir.mkdir(parents=True, exist_ok=True)
        # train all models on the training split
        self._train_all_models()
        # persist all trained models to disk
        self._save_all_models()
        # evaluate all models on the held-out test split
        self.model_scores = self._evaluate_all_models()
        # run statistical significance tests to compare model distributions
        self.statistical_results = self._run_statistical_tests()
        # select the best model based on mean PSNR
        self.best_model_name, self.best_model = self._select_best_model()
        # persist results to disk
        self._save_results()
        self._plot_results()

    def _train_all_models(self):
        '''
        Train all candidate models on the training split of the dataset

        Args:
            None

        Returns:
            None
        '''
        print("\n--- Training all models ---")
        for name, model in self._models.items():
            print(f"\nTraining: {name}")
            model.fit(self._train_clean, self._train_watermarked, self._train_masks)

    def _save_all_models(self):
        '''
        Save all trained models to the saved_models subdirectory of the output directory.
        Each model is saved using its own save() method with a sanitised filename.

        Args:
            None

        Returns:
            None
        '''
        print("\n--- Saving all trained models ---")
        ext_map = {'CNN Autoencoder': '.pt'}
        for name, model in self._models.items():
            ext = ext_map.get(name, '.pkl')
            filename = name.lower().replace(' ', '_') + ext
            save_path = self._saved_models_dir / filename
            model.save(str(save_path))

    def _evaluate_all_models(self):
        '''
        Apply each model to every test image and collect PSNR and SSIM scores

        Args:
            None

        Returns:
            dictionary mapping model names to dictionaries of 'psnr' and 'ssim' score lists
        '''
        print("\n--- Evaluating all models on test set ---")
        model_scores = {name: {'psnr': [], 'ssim': []} for name in self._models}
        for i, (watermarked, clean, mask) in enumerate(
            zip(self._test_watermarked, self._test_clean, self._test_masks)
        ):
            for name, model in self._models.items():
                restored = model.remove_watermark(watermarked.copy(), mask.copy())
                metrics = self._metrics_computer.compute_all(clean, restored)
                model_scores[name]['psnr'].append(metrics['psnr'])
                model_scores[name]['ssim'].append(metrics['ssim'])
            print(f"Evaluated test image {i + 1}/{len(self._test_clean)}")
        # print summary of mean scores per model
        print("\n--- Model Performance Summary ---")
        for name, scores in model_scores.items():
            mean_psnr = np.mean(scores['psnr'])
            mean_ssim = np.mean(scores['ssim'])
            print(f"{name}: Mean PSNR = {mean_psnr:.2f} dB, Mean SSIM = {mean_ssim:.4f}")
        return model_scores

    def _run_statistical_tests(self):
        '''
        Run one-way ANOVA across all models followed by pairwise Welch's t-tests
        with Bonferroni correction to identify statistically significant differences in PSNR scores

        Args:
            None

        Returns:
            dictionary with keys:
                'anova': dict containing 'f_stat' and 'p_value' from one-way ANOVA
                'pairwise': list of dicts with 'model_a', 'model_b', 't_stat', 'p_value',
                            'p_corrected', and 'significant' for each model pair
        '''
        print("\n--- Running Statistical Tests ---")
        psnr_scores = [self.model_scores[name]['psnr'] for name in self._models]
        model_names = list(self._models.keys())
        # one-way ANOVA tests whether any model means differ significantly
        f_stat, p_value_anova = stats.f_oneway(*psnr_scores)
        print(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_value_anova:.4f}")
        if p_value_anova < 0.05:
            print("ANOVA result: Significant differences exist between models (p < 0.05)")
        else:
            print("ANOVA result: No significant differences detected between models (p >= 0.05)")
        # pairwise Welch's t-tests with Bonferroni correction for multiple comparisons
        n_comparisons = len(model_names) * (len(model_names) - 1) // 2
        pairwise_results = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                t_stat, p_val = stats.ttest_ind(psnr_scores[i], psnr_scores[j], equal_var=False)
                # Bonferroni correction: multiply p-value by total number of comparisons
                p_corrected = min(p_val * n_comparisons, 1.0)
                significant = p_corrected < 0.05
                pairwise_results.append({
                    'model_a': model_names[i],
                    'model_b': model_names[j],
                    't_stat': t_stat,
                    'p_value': p_val,
                    'p_corrected': p_corrected,
                    'significant': significant
                })
                print(
                    f"  {model_names[i]} vs {model_names[j]}: "
                    f"t = {t_stat:.3f}, p = {p_val:.4f}, "
                    f"p_corrected = {p_corrected:.4f}, significant = {significant}"
                )
        return {
            'anova': {'f_stat': f_stat, 'p_value': p_value_anova},
            'pairwise': pairwise_results
        }

    def _select_best_model(self):
        '''
        Select the model with the highest mean PSNR score on the test set

        Args:
            None

        Returns:
            tuple of (best model name string, best model object)
        '''
        mean_psnr_per_model = {
            name: np.mean(scores['psnr'])
            for name, scores in self.model_scores.items()
        }
        best_name = max(mean_psnr_per_model, key=mean_psnr_per_model.get)
        print(f"\nBest model selected: {best_name} (Mean PSNR: {mean_psnr_per_model[best_name]:.2f} dB)")
        return best_name, self._models[best_name]

    def _save_results(self):
        '''
        Write a plaintext experiment report containing per-model metrics, ANOVA results,
        pairwise t-test results, and the selected best model to the output directory

        Args:
            None

        Returns:
            None
        '''
        results_path = self._output_dir / 'experiment_results.txt'
        with open(results_path, 'w') as f:
            f.write("=== Watermark Removal A/B Testing Experiment Results ===\n\n")
            f.write("--- Model Performance on Test Set ---\n")
            for name, scores in self.model_scores.items():
                f.write(f"{name}:\n")
                f.write(f"  Mean PSNR: {np.mean(scores['psnr']):.2f} dB  (std: {np.std(scores['psnr']):.2f})\n")
                f.write(f"  Mean SSIM: {np.mean(scores['ssim']):.4f}       (std: {np.std(scores['ssim']):.4f})\n")
            f.write("\n--- One-Way ANOVA ---\n")
            f.write(f"F-statistic: {self.statistical_results['anova']['f_stat']:.4f}\n")
            f.write(f"p-value:     {self.statistical_results['anova']['p_value']:.4f}\n")
            f.write("\n--- Pairwise Welch's t-tests (Bonferroni corrected) ---\n")
            for result in self.statistical_results['pairwise']:
                f.write(
                    f"{result['model_a']} vs {result['model_b']}: "
                    f"t = {result['t_stat']:.3f}, p = {result['p_value']:.4f}, "
                    f"p_corrected = {result['p_corrected']:.4f}, "
                    f"significant = {result['significant']}\n"
                )
            f.write(f"\n--- Best Model Selected ---\n{self.best_model_name}\n")
        print(f"Experiment results saved to {results_path}")

    def _plot_results(self):
        '''
        Generate bar charts comparing model PSNR and SSIM scores with error bars and
        save the figure to the output directory. The best model bar is highlighted in orange.

        Args:
            None

        Returns:
            None
        '''
        model_names = list(self.model_scores.keys())
        mean_psnr = [np.mean(self.model_scores[name]['psnr']) for name in model_names]
        std_psnr = [np.std(self.model_scores[name]['psnr']) for name in model_names]
        mean_ssim = [np.mean(self.model_scores[name]['ssim']) for name in model_names]
        std_ssim = [np.std(self.model_scores[name]['ssim']) for name in model_names]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = range(len(model_names))
        best_idx = model_names.index(self.best_model_name)
        # bar colours: highlight the best model in orange
        psnr_colors = ['darkorange' if i == best_idx else 'steelblue' for i in x]
        ssim_colors = ['darkorange' if i == best_idx else 'steelblue' for i in x]
        # PSNR comparison bar chart
        axes[0].bar(x, mean_psnr, yerr=std_psnr, capsize=5, color=psnr_colors, alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=25, ha='right')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('Model Comparison: PSNR (higher is better)')
        axes[0].grid(axis='y', alpha=0.3)
        # SSIM comparison bar chart
        axes[1].bar(x, mean_ssim, yerr=std_ssim, capsize=5, color=ssim_colors, alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=25, ha='right')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('Model Comparison: SSIM (higher is better)')
        axes[1].grid(axis='y', alpha=0.3)
        plt.suptitle(f'Best Model: {self.best_model_name} (orange)', fontsize=12, y=1.01)
        plt.tight_layout()
        plot_path = self._output_dir / 'model_comparison.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {plot_path}")
