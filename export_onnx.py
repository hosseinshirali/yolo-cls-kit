#!/usr/bin/env python3
"""
ONNX Export Utility for YOLO Classification Models.

Exports a trained YOLO classification model (.pt) to ONNX format for
deployment in production environments (e.g., ONNX Runtime, TensorRT,
OpenVINO, web inference).

Usage:
    # Basic export (default settings)
    python export_onnx.py --model runs/train_results/weights/best.pt

    # Custom image size and output path
    python export_onnx.py --model best.pt --imgsz 224 --output ./exported/model.onnx

    # Export with dynamic batch size (for variable batch inference)
    python export_onnx.py --model best.pt --dynamic

    # Export with FP16 half precision (smaller model, faster inference)
    python export_onnx.py --model best.pt --half

    # Simplify the ONNX graph (requires onnxsim)
    python export_onnx.py --model best.pt --simplify

    # Quick verification after export
    python export_onnx.py --model best.pt --verify
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export a trained YOLO classification model to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_onnx.py --model best.pt
  python export_onnx.py --model best.pt --imgsz 224 --half
  python export_onnx.py --model best.pt --dynamic --simplify --verify
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained YOLO .pt model file (e.g., runs/weights/best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for the exported model (default: 640).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the .onnx file. Default: same directory as the input model.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 half precision (requires CUDA).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes (variable batch size).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the ONNX graph using onnxsim.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (default: auto-selected by Ultralytics).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a quick inference check on the exported ONNX model.",
    )
    return parser.parse_args()


def export_to_onnx(args: argparse.Namespace) -> Path:
    """
    Export a YOLO model to ONNX format using the Ultralytics API.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Path to the exported ONNX file.
    """
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}")
    model = YOLO(str(model_path))

    # Build export kwargs
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    logger.info(f"Exporting to ONNX with settings: {export_kwargs}")
    exported_path = model.export(**export_kwargs)
    exported_path = Path(exported_path)

    # Move to custom output path if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.rename(output_path)
        exported_path = output_path

    logger.info(f"ONNX model saved to: {exported_path}")
    logger.info(f"File size: {exported_path.stat().st_size / (1024 * 1024):.1f} MB")
    return exported_path


def verify_onnx(onnx_path: Path, imgsz: int = 640) -> None:
    """
    Run a quick sanity check on the exported ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        imgsz: Image size used during export.
    """
    try:
        import onnx

        logger.info("Checking ONNX model validity...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure is valid.")
    except ImportError:
        logger.warning("Install 'onnx' package for model validation: pip install onnx")
        return
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return

    try:
        import onnxruntime as ort
        import numpy as np

        logger.info("Running test inference with ONNX Runtime...")
        session = ort.InferenceSession(str(onnx_path))
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape

        # Build a dummy input matching the model's expected shape
        dummy_shape = [
            1 if isinstance(d, str) else d for d in input_shape
        ]
        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

        outputs = session.run(None, {input_meta.name: dummy_input})
        logger.info(
            f"Inference OK — output shape: {outputs[0].shape}, "
            f"num classes: {outputs[0].shape[-1]}"
        )
    except ImportError:
        logger.warning(
            "Install 'onnxruntime' for inference verification: "
            "pip install onnxruntime  (or onnxruntime-gpu)"
        )
    except Exception as e:
        logger.error(f"ONNX Runtime inference check failed: {e}")


def main() -> None:
    """Entry point."""
    args = parse_args()
    onnx_path = export_to_onnx(args)

    if args.verify:
        verify_onnx(onnx_path, imgsz=args.imgsz)

    logger.info("Done! Your model is ready for deployment.")


if __name__ == "__main__":
    main()
