#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:full:1
#SBATCH --mail-user=hossein.shirali@kit.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --partition=normal
#SBATCH --ntasks=1

cd /hkfs/home/haicore/iai/es4994/yolo_classification/

source activate yolo_cls_env

# Verify environment and GPU access
echo "=== Environment Check ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo "========================="


# python run_pipeline.py --config config_yolov8l_auto.yaml

# python run_pipeline.py --config config_yolov8m_auto.yaml

# python run_pipeline.py --config config_yolov8s_auto.yaml

# python run_pipeline.py --config config_yolov8n_auto.yaml

# python run_pipeline.py --config config_yolo11x_auto.yaml

# python run_pipeline.py --config config_yolo11l_auto.yaml

python run_pipeline.py --config config_yolo11m_auto.yaml

# python run_pipeline.py --config config_yolo11s_auto.yaml

# python run_pipeline.py --config config_yolo11n_auto.yaml
