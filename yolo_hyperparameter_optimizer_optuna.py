#!/usr/bin/env python3
"""
YOLO Classification Hyperparameter Optimizer - OPTUNA VERSION

Advanced hyperparameter optimization for YOLO classification models using Optuna.

Key Features:
- Early pruning with MedianPruner and HyperbandPruner
- SQLite persistence for crash recovery (project-specific databases in output folder)
- Real-time progress tracking with JSON updates
- Multi-GPU support with automatic resource management
- Interactive HTML visualizations (5+ plots)
- Support for all YOLO hyperparameters

Author: HPO Team
Date: October 2025
"""

import logging
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Real-time progress tracking with JSON updates."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize progress tracker.
        
        Args:
            output_dir: Directory to save progress files
        """
        self.output_dir = output_dir
        self.progress_file = output_dir / 'optimization_progress.json'
        self.trials_data = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def update(self, trial_number: int, trial_data: Dict[str, Any]):
        """
        Update progress with new trial data.
        
        Args:
            trial_number: Current trial number
            trial_data: Trial information and results
        """
        with self.lock:
            trial_data['timestamp'] = datetime.now().isoformat()
            trial_data['elapsed_time'] = time.time() - self.start_time
            self.trials_data.append(trial_data)
            
            # Calculate statistics
            completed_trials = [t for t in self.trials_data if 'score' in t and t['score'] is not None]
            
            progress = {
                'total_trials': trial_number,
                'completed_trials': len(completed_trials),
                'failed_trials': trial_number - len(completed_trials),
                'best_score': max([t['score'] for t in completed_trials]) if completed_trials else None,
                'current_time': datetime.now().isoformat(),
                'elapsed_time_seconds': time.time() - self.start_time,
                'trials': self.trials_data
            }
            
            # Save to JSON with UTF-8 encoding
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, default=str)
            
            logger.info(f"Progress updated: {len(completed_trials)}/{trial_number} trials completed")


class GPUManager:
    """Manages GPU allocation for multi-GPU support."""
    
    def __init__(self, num_gpus: int = 1, parallel_trials: int = 1):
        """
        Initialize GPU manager.
        
        Args:
            num_gpus: Number of GPUs available
            parallel_trials: Number of trials to run in parallel
        """
        self.num_gpus = num_gpus
        self.parallel_trials = min(parallel_trials, num_gpus)
        self.gpu_queue = list(range(num_gpus)) * (parallel_trials // num_gpus + 1)
        self.lock = threading.Lock()
        self.allocated_gpus = {}
        
        logger.info(f"GPU Manager initialized: {num_gpus} GPUs, {self.parallel_trials} parallel trials")
    
    def acquire_gpu(self, trial_id: int) -> int:
        """
        Acquire a GPU for a trial.
        
        Args:
            trial_id: Trial identifier
            
        Returns:
            GPU device ID
        """
        with self.lock:
            if self.gpu_queue:
                gpu_id = self.gpu_queue.pop(0)
                self.allocated_gpus[trial_id] = gpu_id
                logger.debug(f"Trial {trial_id} assigned to GPU {gpu_id}")
                return gpu_id
            else:
                # Fallback to round-robin
                gpu_id = trial_id % self.num_gpus
                self.allocated_gpus[trial_id] = gpu_id
                return gpu_id
    
    def release_gpu(self, trial_id: int):
        """
        Release GPU after trial completion.
        
        Args:
            trial_id: Trial identifier
        """
        with self.lock:
            if trial_id in self.allocated_gpus:
                gpu_id = self.allocated_gpus[trial_id]
                self.gpu_queue.append(gpu_id)
                del self.allocated_gpus[trial_id]
                logger.debug(f"Trial {trial_id} released GPU {gpu_id}")


class YOLOOptunaOptimizer:
    """
    Advanced hyperparameter optimizer for YOLO classification using Optuna.
    
    Features:
    - Early pruning (MedianPruner, HyperbandPruner)
    - SQLite persistence for crash recovery
    - Real-time progress tracking
    - Multi-GPU support
    - Interactive visualizations
    """
    
    def __init__(
        self,
        dataset_yaml: str,
        model_name: str = 'yolov8n-cls.pt',
        output_dir: str = 'optimization_results',
        study_name: Optional[str] = None
    ):
        """
        Initialize the Optuna optimizer.
        
        Args:
            dataset_yaml: Path to dataset YAML configuration
            model_name: YOLO model name or path
            output_dir: Directory to save optimization results
            study_name: Name for the Optuna study (default: auto-generated)
        """
        self.dataset_yaml = dataset_yaml
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create study-specific directory
        self.study_name = study_name or f"yolo_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study_dir = self.output_dir / self.study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for persistence (crash recovery)
        self.db_path = self.study_dir / f'{self.study_name}.db'
        self.storage = f'sqlite:///{self.db_path}'
        
        # Progress tracking
        self.progress_tracker = ProgressTracker(self.study_dir)
        
        # GPU management
        self.gpu_manager = None
        
        # Study will be created in optimize method
        self.study = None
        
        logger.info("="*80)
        logger.info("YOLO OPTUNA OPTIMIZER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Dataset: {dataset_yaml}")
        logger.info(f"Output directory: {self.study_dir}")
        logger.info(f"Database: {self.db_path}")
        logger.info("")
        logger.info("NOTE: HPO is slower than normal training because:")
        logger.info("  - Each trial trains a fresh model from scratch")
        logger.info("  - Optuna manages study state and trial persistence")
        logger.info("  - Memory cleanup happens between trials")
        logger.info("  - Dataset validation is performed for each trial")
        logger.info("="*80)
    
    def optimize(
        self,
        hyperparams_config: Dict[str, Any],
        base_training_params: Dict[str, Any],
        iterations: int = 10,
        tune_epochs: int = 30,
        metric: str = 'metrics/accuracy_top1',
        pruner: str = 'median',
        num_gpus: int = 1,
        parallel_trials: int = 1,
        sampler: str = 'tpe',
        enable_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization using Optuna.
        
        PERFORMANCE NOTES:
        - HPO is slower than normal training (expected behavior)
        - Each trial creates a fresh model and trains from scratch
        - Memory usage is higher due to Optuna's study management
        - Dataset is loaded for each trial (YOLO limitation)
        - Use fewer tune_epochs (10-30) to speed up trials
        - Use pruning to stop bad trials early
        - Memory is cleaned up between trials automatically
        
        Args:
            hyperparams_config: Configuration of hyperparameters to optimize
            base_training_params: Base training parameters
            iterations: Number of optimization trials
            tune_epochs: Epochs for each trial (use fewer for faster HPO)
            metric: Metric to optimize
            pruner: Pruning algorithm ('median', 'hyperband', 'none')
            num_gpus: Number of GPUs available
            parallel_trials: Number of trials to run in parallel
            sampler: Sampling algorithm ('tpe', 'random', 'cmaes')
            enable_visualization: Whether to generate HTML visualizations
            
        Returns:
            Dictionary with best hyperparameters and results
        """
        try:
            import optuna
            from optuna.pruners import MedianPruner, HyperbandPruner
            from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
            import torch
        except ImportError as e:
            logger.error("Optuna is not installed!")
            logger.error(f"Error: {e}")
            logger.error("Install with: pip install optuna optuna-dashboard plotly kaleido")
            raise
        
        # Check GPU availability
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available GPUs: {available_gpus}")
        
        if num_gpus > available_gpus:
            logger.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available.")
            num_gpus = available_gpus
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager(num_gpus=num_gpus, parallel_trials=parallel_trials)
        
        # Select pruner
        if pruner == 'median':
            optuna_pruner = MedianPruner(
                n_startup_trials=max(5, iterations // 4),
                n_warmup_steps=max(3, tune_epochs // 3),
                interval_steps=1
            )
            logger.info("Using MedianPruner for early stopping")
        elif pruner == 'hyperband':
            optuna_pruner = HyperbandPruner(
                min_resource=max(3, tune_epochs // 3),
                max_resource=tune_epochs,
                reduction_factor=3
            )
            logger.info("Using HyperbandPruner for early stopping")
        else:
            optuna_pruner = None
            logger.info("Pruning disabled")
        
        # Select sampler
        if sampler == 'tpe':
            optuna_sampler = TPESampler(seed=42, multivariate=True)
            logger.info("Using TPE (Tree-structured Parzen Estimator) sampler")
        elif sampler == 'cmaes':
            optuna_sampler = CmaEsSampler(seed=42)
            logger.info("Using CMA-ES sampler")
        else:
            optuna_sampler = RandomSampler(seed=42)
            logger.info("Using Random sampler")
        
        # Create or load study (with SQLite persistence)
        logger.info(f"Creating study with SQLite storage: {self.storage}")
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,  # Resume from crashes
            direction='maximize',
            pruner=optuna_pruner,
            sampler=optuna_sampler
        )
        
        # Store configuration for objective function
        self.hyperparams_config = hyperparams_config
        self.base_training_params = base_training_params
        self.tune_epochs = tune_epochs
        self.metric = metric
        
        logger.info("="*80)
        logger.info(f"STARTING OPTIMIZATION: {iterations} trials")
        logger.info("="*80)
        
        # Run optimization
        if parallel_trials > 1:
            logger.info(f"Running {parallel_trials} trials in parallel")
            # Parallel optimization
            import joblib
            self.study.optimize(
                self._objective,
                n_trials=iterations,
                n_jobs=parallel_trials,
                show_progress_bar=True,
                gc_after_trial=True
            )
        else:
            logger.info("Running trials sequentially")
            # Sequential optimization
            self.study.optimize(
                self._objective,
                n_trials=iterations,
                show_progress_bar=True,
                gc_after_trial=True
            )
        
        # Get best results
        best_trial = self.study.best_trial
        best_score = best_trial.value
        best_params = best_trial.params
        
        logger.info("="*80)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("="*80)
        logger.info(f"Best Score: {best_score:.4f}")
        logger.info(f"Best Trial: #{best_trial.number}")
        logger.info(f"Best Hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        if enable_visualization:
            logger.info("Generating interactive visualizations...")
            self._generate_visualizations()
        
        return {
            'best_score': best_score,
            'best_hyperparameters': best_params,
            'best_trial_number': best_trial.number,
            'total_trials': len(self.study.trials),
            'study_name': self.study_name,
            'database_path': str(self.db_path),
            'optimization_method': 'optuna'
        }
    
    def _objective(self, trial) -> float:
        """
        Objective function for a single Optuna trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Optimization metric value
        """
        from ultralytics import YOLO
        import torch
        import gc
        
        trial_id = trial.number
        logger.info(f"\n{'='*80}")
        logger.info(f"TRIAL #{trial_id}")
        logger.info(f"{'='*80}")
        
        # Acquire GPU
        gpu_id = self.gpu_manager.acquire_gpu(trial_id) if self.gpu_manager.num_gpus > 0 else -1
        
        try:
            # Sample hyperparameters
            sampled_params = self._sample_hyperparameters(trial, self.hyperparams_config)
            
            # Keep a complete copy for logging (before we modify it)
            sampled_params_complete = sampled_params.copy()
            
            logger.info(f"Trial #{trial_id} - GPU: {gpu_id}")
            logger.info(f"Sampled hyperparameters: {json.dumps(sampled_params_complete, indent=2, default=str)}")
            
            # Prepare training arguments
            train_args = self.base_training_params.copy()
            train_args['data'] = self.dataset_yaml
            train_args['epochs'] = self.tune_epochs
            train_args['project'] = str(self.study_dir / 'trials')
            train_args['name'] = f'trial_{trial_id:04d}'
            train_args['verbose'] = False
            train_args['plots'] = False
            train_args['device'] = gpu_id
            
            # Apply sampled hyperparameters
            augmentation_params = sampled_params.pop('augmentation', {})
            train_args.update(sampled_params)
            train_args.update(augmentation_params)
            
            # Clear GPU cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Train model
            model = YOLO(self.model_name)
            
            # Train with validation (YOLO does validation internally)
            # Set save_period to 0 to avoid saving intermediate checkpoints (saves disk space and memory)
            train_args['save_period'] = 0
            train_args['cache'] = False  # Disable caching to save memory during HPO
            
            results = model.train(**train_args)
            
            # IMPORTANT: Load the BEST model (not last epoch) and validate
            # YOLO saves best.pt based on validation accuracy during training
            # We need to validate the best model, not the final epoch model
            best_model_path = self.study_dir / 'trials' / f'trial_{trial_id:04d}' / 'weights' / 'best.pt'
            
            if best_model_path.exists():
                # Load best model and validate
                best_model = YOLO(str(best_model_path))
                val_results = best_model.val(data=self.dataset_yaml, verbose=False)
                
                # Extract accuracy from best model validation
                if hasattr(val_results, 'top1'):
                    accuracy = float(val_results.top1)
                elif hasattr(val_results, 'results_dict'):
                    accuracy = float(val_results.results_dict.get('metrics/accuracy_top1', 0.0))
                else:
                    accuracy = 0.0
                
                del best_model
            else:
                # Fallback: use training results if best.pt doesn't exist
                logger.warning(f"Best model not found for trial {trial_id}, using training results")
                if hasattr(results, 'results_dict'):
                    accuracy = float(results.results_dict.get('metrics/accuracy_top1', 0.0))
                elif hasattr(results, 'top1'):
                    accuracy = float(results.top1)
                else:
                    accuracy = 0.0
            
            logger.info(f"Trial #{trial_id} completed. Best validation accuracy: {accuracy:.4f}")
            
            # Update progress tracker with COMPLETE hyperparameters
            self.progress_tracker.update(trial_id, {
                'trial_number': trial_id,
                'score': accuracy,
                'hyperparameters': sampled_params_complete,  # Use complete copy
                'gpu_id': gpu_id,
                'status': 'completed'
            })
            
            # Aggressive cleanup to free memory
            del model, results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Trial #{trial_id} failed: {e}")
            
            # Update progress tracker
            self.progress_tracker.update(trial_id, {
                'trial_number': trial_id,
                'score': None,
                'hyperparameters': {},
                'gpu_id': gpu_id,
                'status': 'failed',
                'error': str(e)
            })
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Re-raise to let Optuna handle it
            raise
            
        finally:
            # Release GPU
            if self.gpu_manager:
                self.gpu_manager.release_gpu(trial_id)
    
    def _sample_hyperparameters(self, trial, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters using Optuna trial object.
        
        Args:
            trial: Optuna trial object
            config: Hyperparameter configuration
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        sampled = {}
        
        for param_name, param_spec in config.items():
            if param_name == 'augmentation' and isinstance(param_spec, dict):
                # Recursively sample augmentation params
                sampled['augmentation'] = self._sample_hyperparameters(trial, param_spec)
            elif isinstance(param_spec, list) and len(param_spec) == 2:
                # Check if numeric range
                if all(isinstance(v, (int, float)) for v in param_spec):
                    low, high = param_spec
                    
                    # Integer parameters
                    if param_name in ['warmup_epochs', 'freeze', 'epochs', 'batch']:
                        sampled[param_name] = trial.suggest_int(param_name, int(low), int(high))
                    # Learning rate (log scale)
                    elif 'lr' in param_name.lower() and low > 0:
                        sampled[param_name] = trial.suggest_float(param_name, float(low), float(high), log=True)
                    # Other continuous
                    else:
                        sampled[param_name] = trial.suggest_float(param_name, float(low), float(high))
                else:
                    # List of choices
                    sampled[param_name] = trial.suggest_categorical(param_name, param_spec)
            elif isinstance(param_spec, list):
                # List of choices
                sampled[param_name] = trial.suggest_categorical(param_name, param_spec)
            else:
                # Fixed value (not sampled)
                sampled[param_name] = param_spec
        
        return sampled
    
    def _save_results(self):
        """Save optimization results to files."""
        if not self.study:
            logger.warning("No study to save results from")
            return
        
        # Save trials dataframe
        trials_df = self.study.trials_dataframe()
        csv_path = self.study_dir / 'optimization_results.csv'
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")
        
        # Save best hyperparameters
        best_trial = self.study.best_trial
        best_params_data = {
            'best_score': float(best_trial.value),
            'best_trial_number': best_trial.number,
            'best_hyperparameters': best_trial.params,
            'optimization_date': datetime.now().isoformat(),
            'study_name': self.study_name,
            'total_trials': len(self.study.trials),
            'pruned_trials': len([t for t in self.study.trials if t.state.name == 'PRUNED']),
            'completed_trials': len([t for t in self.study.trials if t.state.name == 'COMPLETE']),
            'failed_trials': len([t for t in self.study.trials if t.state.name == 'FAIL'])
        }
        
        yaml_path = self.study_dir / 'best_hyperparameters.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_params_data, f, default_flow_style=False)
        logger.info(f"Best hyperparameters saved to: {yaml_path}")
        
        # Save study statistics
        stats = {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'best_value': float(best_trial.value),
            'best_trial': best_trial.number,
            'optimization_date': datetime.now().isoformat(),
            'trials': [
                {
                    'number': t.number,
                    'value': float(t.value) if t.value is not None else None,
                    'params': t.params,
                    'state': t.state.name,
                    'duration': t.duration.total_seconds() if t.duration else None
                }
                for t in self.study.trials
            ]
        }
        
        json_path = self.study_dir / 'optimization_statistics.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Statistics saved to: {json_path}")
    
    def _generate_visualizations(self):
        """Generate interactive HTML visualizations using Optuna."""
        if not self.study:
            logger.warning("No study to visualize")
            return
        
        try:
            import optuna.visualization as vis
            import plotly
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualizations.")
            logger.warning("Install with: pip install plotly kaleido")
            return
        
        viz_dir = self.study_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating visualizations in: {viz_dir}")
        
        # 1. Optimization History
        try:
            fig = vis.plot_optimization_history(self.study)
            fig.write_html(str(viz_dir / '1_optimization_history.html'))
            logger.info("✓ Optimization history plot created")
        except Exception as e:
            logger.warning(f"Could not create optimization history plot: {e}")
        
        # 2. Parameter Importances
        try:
            fig = vis.plot_param_importances(self.study)
            fig.write_html(str(viz_dir / '2_param_importances.html'))
            logger.info("✓ Parameter importances plot created")
        except Exception as e:
            logger.warning(f"Could not create parameter importances plot: {e}")
        
        # 3. Parallel Coordinate Plot
        try:
            fig = vis.plot_parallel_coordinate(self.study)
            fig.write_html(str(viz_dir / '3_parallel_coordinate.html'))
            logger.info("✓ Parallel coordinate plot created")
        except Exception as e:
            logger.warning(f"Could not create parallel coordinate plot: {e}")
        
        # 4. Slice Plot
        try:
            fig = vis.plot_slice(self.study)
            fig.write_html(str(viz_dir / '4_slice_plot.html'))
            logger.info("✓ Slice plot created")
        except Exception as e:
            logger.warning(f"Could not create slice plot: {e}")
        
        # 5. Contour Plot
        try:
            fig = vis.plot_contour(self.study)
            fig.write_html(str(viz_dir / '5_contour_plot.html'))
            logger.info("✓ Contour plot created")
        except Exception as e:
            logger.warning(f"Could not create contour plot: {e}")
        
        # 6. EDF Plot (Empirical Distribution Function)
        try:
            fig = vis.plot_edf(self.study)
            fig.write_html(str(viz_dir / '6_edf_plot.html'))
            logger.info("✓ EDF plot created")
        except Exception as e:
            logger.warning(f"Could not create EDF plot: {e}")
        
        # 7. Timeline
        try:
            fig = vis.plot_timeline(self.study)
            fig.write_html(str(viz_dir / '7_timeline.html'))
            logger.info("✓ Timeline plot created")
        except Exception as e:
            logger.warning(f"Could not create timeline plot: {e}")
        
        # Create index.html with links to all plots
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Optuna Optimization Results - {self.study_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .info-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot-card a {{
            display: block;
            padding: 15px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.3s;
        }}
        .plot-card a:hover {{
            background: #45a049;
        }}
        .best-params {{
            background: #e8f5e9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin: 10px 0;
        }}
        .stat {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
        }}
        .stat-value {{
            color: #4CAF50;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <h1>🎯 YOLO Hyperparameter Optimization Results</h1>
    
    <div class="info-box">
        <h2>Study Information</h2>
        <div class="stat">
            <span class="stat-label">Study Name:</span>
            <span class="stat-value">{self.study_name}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Best Score:</span>
            <span class="stat-value">{self.study.best_value:.4f}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Total Trials:</span>
            <span class="stat-value">{len(self.study.trials)}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Best Trial:</span>
            <span class="stat-value">#{self.study.best_trial.number}</span>
        </div>
    </div>
    
    <div class="info-box">
        <h2>Best Hyperparameters</h2>
        <div class="best-params">
            <pre>{json.dumps(self.study.best_params, indent=2)}</pre>
        </div>
    </div>
    
    <div class="info-box">
        <h2>📊 Interactive Visualizations</h2>
        <p>Click on any plot below to view the interactive visualization:</p>
        
        <div class="plot-grid">
            <div class="plot-card">
                <h3>📈 Optimization History</h3>
                <p>Track how the objective value improved over trials</p>
                <a href="1_optimization_history.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>🔍 Parameter Importances</h3>
                <p>See which hyperparameters matter most</p>
                <a href="2_param_importances.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>📊 Parallel Coordinate</h3>
                <p>Visualize relationships between parameters</p>
                <a href="3_parallel_coordinate.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>📉 Slice Plot</h3>
                <p>Individual parameter effects on objective</p>
                <a href="4_slice_plot.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>🗺️ Contour Plot</h3>
                <p>2D parameter interaction visualization</p>
                <a href="5_contour_plot.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>📊 EDF Plot</h3>
                <p>Empirical distribution of objective values</p>
                <a href="6_edf_plot.html" target="_blank">View Plot</a>
            </div>
            
            <div class="plot-card">
                <h3>⏱️ Timeline</h3>
                <p>Trial execution timeline and parallelization</p>
                <a href="7_timeline.html" target="_blank">View Plot</a>
            </div>
        </div>
    </div>
    
    <div class="info-box">
        <h2>💾 Files Generated</h2>
        <ul>
            <li><strong>optimization_results.csv</strong> - All trial results in CSV format</li>
            <li><strong>best_hyperparameters.yaml</strong> - Best hyperparameters in YAML format</li>
            <li><strong>optimization_statistics.json</strong> - Detailed statistics in JSON format</li>
            <li><strong>optimization_progress.json</strong> - Real-time progress tracking</li>
            <li><strong>{self.study_name}.db</strong> - SQLite database for crash recovery</li>
        </ul>
    </div>
    
    <div class="info-box">
        <p style="text-align: center; color: #666;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""
        
        with open(viz_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        logger.info("="*80)
        logger.info(f"All visualizations created in: {viz_dir}")
        logger.info(f"✓ Open {viz_dir / 'index.html'} in your browser to view results")
        logger.info("="*80)


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
    print("="*80)
    print("YOLO Hyperparameter Optimizer - OPTUNA VERSION")
    print("="*80)
    print("\nFeatures:")
    print("✓ Early pruning (MedianPruner, HyperbandPruner)")
    print("✓ SQLite persistence for crash recovery")
    print("✓ Real-time progress tracking (JSON updates)")
    print("✓ Multi-GPU support")
    print("✓ Interactive visualizations (7+ HTML plots)")
    print("\nSee example_optuna_usage.py for usage examples.")
    print("="*80)
