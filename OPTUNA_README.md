# Optuna Hyperparameter Optimization for YOLO Classification

This repository now supports **Optuna** - an advanced hyperparameter optimization framework with powerful features that go beyond Ray Tune.

## 🎯 Why Optuna?

Optuna provides several advantages for hyperparameter optimization:

### ✨ Key Features

1. **🔪 Early Pruning** - Stop unpromising trials early to save computation
   - MedianPruner: Stops trials performing below median
   - HyperbandPruner: Aggressive pruning using successive halving
   
2. **💾 SQLite Persistence** - Crash recovery and resumability
   - All trials automatically saved to SQLite database
   - Resume optimization after crashes or interruptions
   - Project-specific databases in output folder

3. **📊 Real-time Progress Tracking** - JSON updates for monitoring
   - `optimization_progress.json` updates in real-time
   - Track completed/failed trials
   - Monitor best score during optimization

4. **🚀 Multi-GPU Support** - Efficient parallel optimization
   - Automatic GPU resource management
   - Run multiple trials in parallel (1 per GPU)
   - Smart GPU allocation and cleanup

5. **📈 Interactive Visualizations** - 7+ HTML plots
   - Optimization history
   - Parameter importances
   - Parallel coordinate plot
   - Slice plots
   - Contour plots
   - EDF (Empirical Distribution Function)
   - Timeline visualization

## 📦 Installation

Install Optuna and visualization dependencies:

```bash
pip install optuna>=3.0.0 plotly>=5.0.0 kaleido>=0.2.1
```

Optional: Install Optuna Dashboard for web-based monitoring:
```bash
pip install optuna-dashboard>=0.13.0
```

Or install all requirements:
```bash
pip install -r requirements_pipeline.txt
```

## 🚀 Quick Start

### Option 1: Using Python Script

```python
from yolo_hyperparameter_optimizer_optuna import YOLOOptunaOptimizer

# Initialize optimizer
optimizer = YOLOOptunaOptimizer(
    dataset_yaml='dataset/smnk/data.yaml',
    model_name='yolov8n-cls.pt',
    output_dir='output_dir/orient',
    study_name='yolov8_orient_optuna'  # Optional: name your study
)

# Define hyperparameters to optimize
hyperparams_to_optimize = {
    'optimizer': ['SGD', 'Adam', 'AdamW'],
    'lr0': [0.0001, 0.01],
    'momentum': [0.8, 0.95],
    'weight_decay': [0.0001, 0.001],
    'augmentation': {
        'hsv_h': [0.0, 0.02],
        'hsv_s': [0.3, 0.7],
        'degrees': [0, 15],
        'fliplr': [0.3, 0.7],
    }
}

# Base training parameters
base_training_params = {
    'imgsz': 224,
    'batch': 16,
    'workers': 8,
    'patience': 10,
}

# Run optimization
results = optimizer.optimize(
    hyperparams_config=hyperparams_to_optimize,
    base_training_params=base_training_params,
    iterations=20,           # Number of trials
    tune_epochs=30,          # Epochs per trial
    pruner='median',         # Early pruning strategy
    sampler='tpe',           # Bayesian optimization
    num_gpus=1,
    parallel_trials=1,       # Sequential for single GPU
    enable_visualization=True
)
```

### Option 2: Using YAML Config

Update your `config_example.yaml`:

```yaml
optimization:
  enabled: true
  iterations: 20
  
  # Enable Optuna
  use_optuna: true
  use_ray: false
  
  # Optuna settings
  study_name: 'yolov8_orient_optuna'
  pruner: 'median'      # 'median', 'hyperband', 'none'
  sampler: 'tpe'        # 'tpe', 'random', 'cmaes'
  parallel_trials: 1
  num_gpus: 1
  tune_epochs: 30
  enable_visualization: true
  
  hyperparameters:
    optimizer: ['SGD', 'Adam', 'AdamW']
    lr0: [0.0001, 0.01]
    momentum: [0.8, 0.95]
    # ... more parameters
```

Then run:
```bash
python run_pipeline.py --config config_example.yaml
```

## 🎨 Visualization Gallery

After optimization completes, you'll find 7+ interactive HTML visualizations in:
```
output_dir/<study_name>/visualizations/
```

Open `index.html` in your browser to see all plots:

1. **Optimization History** - Track improvement over trials
2. **Parameter Importances** - Which hyperparameters matter most
3. **Parallel Coordinate** - Multi-dimensional parameter relationships
4. **Slice Plot** - Individual parameter effects
5. **Contour Plot** - 2D parameter interactions
6. **EDF Plot** - Distribution of objective values
7. **Timeline** - Trial execution timeline and parallelization

## 🔧 Advanced Features

### 1. Crash Recovery

Optuna automatically saves all trials to SQLite. If optimization crashes or is interrupted:

```python
# Same study_name will automatically resume!
optimizer = YOLOOptunaOptimizer(
    dataset_yaml='dataset/smnk/data.yaml',
    model_name='yolov8n-cls.pt',
    output_dir='output_dir/orient',
    study_name='yolov8_orient_optuna'  # Same name as before
)

# This will continue from where it left off
results = optimizer.optimize(...)
```

The database is located at:
```
output_dir/<study_name>/<study_name>.db
```

### 2. Real-time Monitoring

Monitor optimization progress in real-time:

```bash
# Watch progress file
tail -f output_dir/<study_name>/optimization_progress.json
```

Or use Optuna Dashboard:
```bash
optuna-dashboard sqlite:///output_dir/<study_name>/<study_name>.db
```
Then open http://localhost:8080 in your browser.

### 3. Multi-GPU Parallel Optimization

For faster optimization with multiple GPUs:

```python
results = optimizer.optimize(
    hyperparams_config=hyperparams_to_optimize,
    base_training_params=base_training_params,
    iterations=20,
    tune_epochs=20,
    num_gpus=2,              # 2 GPUs available
    parallel_trials=2,       # Run 2 trials simultaneously
    pruner='hyperband',      # Aggressive pruning for speed
)
```

### 4. Pruning Strategies

Choose the right pruning strategy for your use case:

**MedianPruner** (Recommended for most cases)
```python
pruner='median'
```
- Stops trials performing below median of completed trials
- Conservative - won't prune too aggressively
- Good for balanced exploration/exploitation

**HyperbandPruner** (For aggressive speed-up)
```python
pruner='hyperband'
```
- Uses successive halving for aggressive pruning
- Faster optimization but may miss good configurations
- Good when you have many trials to run

**No Pruning**
```python
pruner='none'
```
- Let all trials complete fully
- More accurate but slower
- Use when trials are fast

### 5. Sampling Algorithms

**TPE (Tree-structured Parzen Estimator)** - Recommended
```python
sampler='tpe'
```
- Smart Bayesian optimization
- Learns from previous trials
- Best for most use cases

**Random Sampling**
```python
sampler='random'
```
- Pure random search
- Good baseline, no assumptions
- Use for comparison

**CMA-ES**
```python
sampler='cmaes'
```
- Evolutionary strategy
- Good for continuous parameters
- Can be slower than TPE

## 📁 Output Files

After optimization, you'll find these files in `output_dir/<study_name>/`:

```
<study_name>/
├── <study_name>.db                    # SQLite database (crash recovery)
├── optimization_results.csv           # All trials in CSV format
├── best_hyperparameters.yaml          # Best parameters in YAML
├── optimization_statistics.json       # Detailed statistics
├── optimization_progress.json         # Real-time progress tracking
├── trials/                            # Individual trial results
│   ├── trial_0001/
│   ├── trial_0002/
│   └── ...
└── visualizations/                    # Interactive HTML plots
    ├── index.html                     # Main visualization dashboard
    ├── 1_optimization_history.html
    ├── 2_param_importances.html
    ├── 3_parallel_coordinate.html
    ├── 4_slice_plot.html
    ├── 5_contour_plot.html
    ├── 6_edf_plot.html
    └── 7_timeline.html
```

## 🆚 Optuna vs Ray Tune

| Feature | Optuna | Ray Tune |
|---------|--------|----------|
| Early Pruning | ✅ Built-in (Median, Hyperband) | ✅ ASHA Scheduler |
| Persistence | ✅ SQLite (automatic) | ❌ Manual checkpointing |
| Crash Recovery | ✅ Automatic resume | ⚠️ Manual setup required |
| Progress Tracking | ✅ JSON + Dashboard | ⚠️ Ray Dashboard (complex) |
| Visualizations | ✅ 7+ interactive HTML plots | ⚠️ Limited |
| Multi-GPU | ✅ Built-in resource manager | ✅ Ray resources |
| Bayesian Optimization | ✅ TPE sampler | ✅ Multiple samplers |
| Ease of Use | ✅ Simple API | ⚠️ More complex setup |
| Database | ✅ SQLite (portable) | ❌ None |

**When to use Optuna:**
- ✅ You want crash recovery without manual setup
- ✅ You need rich visualizations
- ✅ You want simple, clean code
- ✅ You're running long optimizations that might be interrupted
- ✅ You want to analyze results with interactive plots

**When to use Ray Tune:**
- ✅ You're already using Ray for distributed computing
- ✅ You need distributed optimization across multiple machines
- ✅ You have existing Ray infrastructure

## 📚 Examples

See `example_optuna_usage.py` for complete examples:

```bash
python example_optuna_usage.py
```

Examples include:
- Basic optimization
- Multi-GPU parallel optimization
- Crash recovery demonstration

## 🐛 Troubleshooting

### "Import optuna could not be resolved"

Install Optuna:
```bash
pip install optuna plotly kaleido
```

### Visualizations not generating

Install visualization dependencies:
```bash
pip install plotly kaleido
```

### Out of Memory (OOM) errors

1. Reduce batch size in `base_training_params`
2. Reduce `tune_epochs`
3. Use `parallel_trials=1` for sequential execution
4. Enable aggressive pruning: `pruner='hyperband'`

### Database locked error

Only one process can write to the SQLite database at a time. Make sure:
- Only one optimization is running per study
- Close Optuna Dashboard before running new optimization

## 📄 License

Same as the main repository.

## 🤝 Contributing

Contributions welcome! Please test your changes with:
```bash
python example_optuna_usage.py
```

## 📞 Support

For issues or questions:
1. Check this README
2. See `example_optuna_usage.py` for working examples
3. Review Optuna documentation: https://optuna.readthedocs.io/
