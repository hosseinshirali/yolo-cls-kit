#!/usr/bin/env python3
"""
YOLO Classification Hyperparameter Optimizer - SIMPLIFIED VERSION

This version follows Ultralytics' official Ray Tune guide for proper resource management.
Key principle: Keep it simple, let Ray Tune handle resource management.

Author: Refactored based on Ultralytics best practices
Date: October 2025
"""

import logging
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class YOLOHyperparameterOptimizer:
    """
    Simplified hyperparameter optimizer for YOLO classification models.
    Follows Ultralytics Ray Tune best practices.
    """
    
    def __init__(
        self,
        dataset_yaml: str,
        model_name: str = 'yolov8n-cls.pt',
        output_dir: str = 'optimization_results',
        use_ray: bool = False
    ):
        """
        Initialize the optimizer.
        
        Args:
            dataset_yaml: Path to dataset YAML configuration
            model_name: YOLO model name or path
            output_dir: Directory to save optimization results
            use_ray: Whether to use Ray Tune (requires installation)
        """
        self.dataset_yaml = dataset_yaml
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ray = use_ray
        
        # Results tracking
        self.results = []
        self.best_params = None
        self.best_score = -float('inf')
        
        logger.info(f"Initialized optimizer for model: {model_name}")
        logger.info(f"Dataset config: {dataset_yaml}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using Ray Tune: {use_ray}")
    
    def optimize(
        self,
        hyperparams_config: Dict[str, Any],
        base_training_params: Dict[str, Any],
        iterations: int = 10,
        tune_epochs: int = 30,
        strategy: str = 'random',
        metric: str = 'metrics/accuracy_top1'
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization without Ray Tune (simple random/grid search).
        
        Args:
            hyperparams_config: Configuration of hyperparameters to optimize
            base_training_params: Base training parameters
            iterations: Number of optimization iterations
            tune_epochs: Epochs for each trial
            strategy: 'random' or 'grid' (only random supported for now)
            metric: Metric to optimize
            
        Returns:
            Dictionary with best hyperparameters and results
        """
        import numpy as np
        from ultralytics import YOLO
        
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION STARTED (Random Search)")
        logger.info("="*80)
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Iterations: {iterations}")
        logger.info(f"Epochs per trial: {tune_epochs}")

        self.results = []
        self.best_params = None
        self.best_score = -float('inf')

        for i in range(iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"TRIAL {i+1}/{iterations}")
            logger.info(f"{'='*80}")
            
            # Sample hyperparameters
            sampled_params = self._sample_hyperparameters(hyperparams_config)
            trial_name = f"trial_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Sampled hyperparameters: {json.dumps(sampled_params, indent=2, default=str)}")
            
            # Prepare training args
            train_args = base_training_params.copy()
            train_args['data'] = self.dataset_yaml
            train_args['epochs'] = tune_epochs
            train_args['project'] = str(self.output_dir)
            train_args['name'] = trial_name
            train_args['verbose'] = False
            train_args['plots'] = False
            
            # Apply sampled hyperparameters
            augmentation_params = sampled_params.pop('augmentation', {})
            train_args.update(sampled_params)
            train_args.update(augmentation_params)
            
            try:
                # Train
                model = YOLO(self.model_name)
                model.train(**train_args)
                
                # Validate
                val_results = model.val(data=self.dataset_yaml)
                
                # Extract accuracy
                if hasattr(val_results, 'top1'):
                    accuracy = float(val_results.top1)
                elif hasattr(val_results, 'results_dict'):
                    accuracy = float(val_results.results_dict.get('metrics/accuracy_top1', 0.0))
                else:
                    accuracy = 0.0
                
                logger.info(f"Trial {trial_name} completed. Accuracy: {accuracy:.4f}")
                
                # Track results
                self.results.append({
                    'trial': i + 1,
                    'score': accuracy,
                    'hyperparameters': sampled_params,
                    'metrics': {'accuracy_top1': accuracy}
                })
                
                # Update best
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_params = sampled_params.copy()
                    logger.info(f"✓ New best score: {accuracy:.4f}")
                    
            except Exception as e:
                logger.error(f"Trial {trial_name} failed: {e}")
                self.results.append({
                    'trial': i + 1,
                    'score': 0.0,
                    'hyperparameters': sampled_params,
                    'metrics': {'error': str(e)}
                })
        
        # Save results
        self._save_results()
        
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*80)
        logger.info(f"Best Score: {self.best_score:.4f}")
        logger.info(f"Best Hyperparameters: {json.dumps(self.best_params, indent=2, default=str)}")
        
        return {
            'best_score': self.best_score,
            'best_hyperparameters': self.best_params,
            'all_results': self.results
        }
    
    def _sample_hyperparameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters randomly from configuration.
        
        Args:
            config: Hyperparameter configuration
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        import numpy as np
        
        sampled = {}
        
        for param_name, param_spec in config.items():
            if param_name == 'augmentation' and isinstance(param_spec, dict):
                # Recursively sample augmentation params
                sampled['augmentation'] = self._sample_hyperparameters(param_spec)
            elif isinstance(param_spec, list) and len(param_spec) == 2:
                # Check if numeric range
                if all(isinstance(v, (int, float)) for v in param_spec):
                    low, high = param_spec
                    
                    # Integer parameters
                    if param_name in ['warmup_epochs', 'freeze', 'epochs']:
                        sampled[param_name] = int(np.random.uniform(low, high))
                    # Learning rate (log scale)
                    elif 'lr' in param_name.lower():
                        sampled[param_name] = float(10 ** np.random.uniform(np.log10(low), np.log10(high)))
                    # Other continuous
                    else:
                        sampled[param_name] = float(np.random.uniform(low, high))
                else:
                    # List of choices
                    sampled[param_name] = np.random.choice(param_spec).item()
            elif isinstance(param_spec, list):
                # List of choices
                sampled[param_name] = np.random.choice(param_spec).item()
            else:
                # Fixed value
                sampled[param_name] = param_spec
        
        return sampled
    
    def optimize_with_ray(
        self,
        hyperparams_config: Dict[str, Any],
        base_training_params: Dict[str, Any],
        iterations: int = 10,
        tune_epochs: int = 30,
        metric: str = 'metrics/accuracy_top1',
        num_gpus: int = 1
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Ray Tune.
        SIMPLIFIED VERSION following Ultralytics guide.
        
        Args:
            hyperparams_config: Configuration of hyperparameters to optimize
            base_training_params: Base training parameters
            iterations: Number of optimization iterations
            tune_epochs: Epochs for each trial
            metric: Metric to optimize
            num_gpus: Number of GPUs to use (default: 1)
            
        Returns:
            Dictionary with best hyperparameters and results
        """
        logger.info("="*80)
        logger.info("RAY TUNE OPTIMIZATION STARTED (SIMPLIFIED)")
        logger.info("="*80)

        try:
            from ray import tune
            from ray.tune.schedulers import ASHAScheduler
            import torch
        except ImportError as e:
            logger.error("Ray Tune is not installed!")
            logger.error(f"Error: {e}")
            raise

        # Check GPU availability
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available GPUs: {available_gpus}")
        
        if num_gpus > available_gpus:
            logger.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available.")
            num_gpus = available_gpus

        # Build search space following Ultralytics pattern
        search_space = self._build_tune_search_space(hyperparams_config)
        
        # Wrap trainable with resources to ensure only 1 trial per GPU
        from ray.tune import with_resources
        
        # Define trainable function (SIMPLE like the guide!)
        def train_yolo_trial(config):
            """
            One Ray Tune trial - SIMPLE VERSION.
            Just train YOLO, no manual cleanup!
            """
            from ultralytics import YOLO
            import torch
            import gc
            import time
            
            # Clear any previous GPU state at the start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use simple trial name to avoid Windows path length issues
            trial_id = f"trial_{int(time.time())}"
            
            # Load model fresh for this trial
            model = YOLO(self.model_name)
            
            # Prepare training args
            train_args = base_training_params.copy()
            
            # Override with Ray Tune settings
            train_args['data'] = self.dataset_yaml
            train_args['epochs'] = tune_epochs
            train_args['project'] = str(self.output_dir / 'ray_trials')
            train_args['name'] = trial_id  # Simple name to avoid path length issues
            train_args['verbose'] = False
            train_args['plots'] = False
            
            # Apply hyperparameters from Ray Tune
            augmentation_params = config.pop('augmentation', {}) if 'augmentation' in config else {}
            train_args.update(config)
            train_args.update(augmentation_params)
            
            try:
                # Train (YOLO handles everything internally)
                model.train(**train_args)
                
                # Validate
                val_results = model.val(data=self.dataset_yaml)
                
                # Extract accuracy
                if hasattr(val_results, 'top1'):
                    accuracy = float(val_results.top1)
                elif hasattr(val_results, 'results_dict'):
                    accuracy = float(val_results.results_dict.get('metrics/accuracy_top1', 0.0))
                else:
                    accuracy = 0.0
                
                # Report to Ray Tune
                tune.report(**{metric: accuracy})
                
            finally:
                # Clean up this trial's model and free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            metric=metric,
            mode='max',
            max_t=tune_epochs,
            grace_period=min(3, tune_epochs),
            reduction_factor=2
        )
        
        # Configure and run tuner (MODERN API)
        logger.info(f"Starting Ray Tune with {iterations} trials...")
        
        # Use trial name creator to avoid long paths on Windows
        from ray import train
        
        def trial_name_creator(trial):
            """Create short trial names to avoid Windows path length issues"""
            return f"trial_{trial.trial_id[:8]}"
        
        # Wrap trainable with resources - 1 GPU per trial ensures sequential execution
        trainable_with_resources = with_resources(
            train_yolo_trial,
            resources={"cpu": 1, "gpu": 1 if num_gpus > 0 else 0}
        )
        
        tuner = tune.Tuner(
            trainable_with_resources,  # Use wrapped trainable
            param_space=search_space,
            tune_config=tune.TuneConfig(
                # Don't pass metric/mode here since scheduler already has them
                num_samples=iterations,
                scheduler=scheduler,
                trial_name_creator=trial_name_creator,  # Short trial names
                trial_dirname_creator=trial_name_creator,  # Short directory names
                max_concurrent_trials=1 if num_gpus == 1 else num_gpus,  # Force sequential on single GPU
            ),
            run_config=tune.RunConfig(
                name='yolo_hpo',  # Shorter name
                storage_path=str(self.output_dir.resolve() / 'ray_tune'),  # Absolute path required
                verbose=1,
            ),
        )
        
        # Run optimization
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=metric, mode='max')
        
        self.best_score = float(best_result.metrics.get(metric, 0.0))
        self.best_params = dict(best_result.config)
        
        # Collect all results
        self.results = []
        for i, result in enumerate(results, start=1):
            self.results.append({
                'trial': i,
                'score': float(result.metrics.get(metric, 0.0)),
                'hyperparameters': dict(result.config),
                'metrics': dict(result.metrics)
            })
        
        # Save results
        self._save_results()
        
        logger.info("="*80)
        logger.info("RAY TUNE OPTIMIZATION COMPLETED")
        logger.info("="*80)
        logger.info(f"Best Score: {self.best_score:.4f}")
        logger.info(f"Best Hyperparameters: {json.dumps(self.best_params, indent=2)}")
        
        return {
            'best_score': self.best_score,
            'best_hyperparameters': self.best_params,
            'all_results': self.results,
            'optimization_method': 'ray_tune'
        }
    
    def _build_tune_search_space(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Ray Tune search space from hyperparameter config.
        
        Args:
            config: Hyperparameter configuration
            
        Returns:
            Ray Tune search space dictionary
        """
        from ray import tune
        import numpy as np
        
        search_space = {}
        
        for param_name, param_spec in config.items():
            if param_name == 'augmentation' and isinstance(param_spec, dict):
                # Recursively build augmentation space
                search_space['augmentation'] = self._build_tune_search_space(param_spec)
            elif isinstance(param_spec, list) and len(param_spec) == 2:
                # Check if numeric range
                if all(isinstance(v, (int, float)) for v in param_spec):
                    low, high = param_spec
                    
                    # Integer parameters
                    if param_name in ['warmup_epochs', 'freeze', 'epochs']:
                        search_space[param_name] = tune.randint(int(low), int(high) + 1)
                    # Learning rate parameters (log scale) - only if both bounds > 0
                    elif 'lr' in param_name.lower() and low > 0 and high > 0:
                        search_space[param_name] = tune.loguniform(float(low), float(high))
                    # Other continuous parameters (use uniform)
                    else:
                        search_space[param_name] = tune.uniform(float(low), float(high))
                else:
                    # List of choices
                    search_space[param_name] = tune.choice(param_spec)
            elif isinstance(param_spec, list):
                # List of choices
                search_space[param_name] = tune.choice(param_spec)
            else:
                # Fixed value
                search_space[param_name] = param_spec
        
        return search_space
    
    def _save_results(self):
        """Save optimization results to files."""
        # Save all results to CSV
        results_df = pd.DataFrame([
            {
                'trial': r['trial'],
                'score': r['score'],
                **{f'hp_{k}': v for k, v in r['hyperparameters'].items() 
                   if k != 'augmentation' and not isinstance(v, (dict, list))},
            }
            for r in self.results
        ])
        
        csv_path = self.output_dir / 'optimization_results.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save best hyperparameters to YAML
        best_params_path = self.output_dir / 'best_hyperparameters.yaml'
        with open(best_params_path, 'w') as f:
            yaml.dump({
                'best_score': float(self.best_score),
                'best_hyperparameters': self.best_params,
                'optimization_date': datetime.now().isoformat()
            }, f, default_flow_style=False)
        logger.info(f"Best hyperparameters saved to: {best_params_path}")
        
        # Save all results to JSON
        json_path = self.output_dir / 'optimization_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to: {json_path}")


def load_best_hyperparameters(yaml_path: str) -> Dict[str, Any]:
    """
    Load best hyperparameters from saved YAML file.
    
    Args:
        yaml_path: Path to best_hyperparameters.yaml
        
    Returns:
        Dictionary of best hyperparameters
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    logger.info(f"Loaded best hyperparameters from: {yaml_path}")
    logger.info(f"Best score was: {data['best_score']:.4f}")
    
    return data['best_hyperparameters']


def apply_optimized_hyperparameters(
    training_params: Dict[str, Any],
    augmentations: Dict[str, Any],
    best_hyperparams: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply optimized hyperparameters to training parameters and augmentations.
    
    Args:
        training_params: Original training parameters
        augmentations: Original augmentation parameters
        best_hyperparams: Best hyperparameters from optimization
        
    Returns:
        Tuple of (updated_training_params, updated_augmentations)
    """
    updated_training = training_params.copy()
    updated_augmentations = augmentations.copy() if augmentations else {}
    
    # Apply training hyperparameters
    for key, value in best_hyperparams.items():
        if key == 'augmentation':
            # Apply augmentation parameters
            updated_augmentations.update(value)
        else:
            # Apply training parameters
            updated_training[key] = value
    
    logger.info("Applied optimized hyperparameters to training configuration")
    
    return updated_training, updated_augmentations


if __name__ == "__main__":
    print("YOLO Hyperparameter Optimizer (Simplified)")
    print("This version follows Ultralytics Ray Tune best practices")
    print("See the guide for usage examples.")
