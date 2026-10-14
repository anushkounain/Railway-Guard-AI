# Models Directory

This folder is intended for local YOLO model weights.

## Supplying Your Checkpoint

1. Copy your trained checkpoint (e.g., `best.pt` or `best new.pt`) into this directory, OR specify its absolute path using the `MODEL_PATH` environment variable in `.env`.
   
   Example:
   ```env
   MODEL_PATH=models/best.pt
   ```

2. By default, model weights (`*.pt`, `*.onnx`, etc.) are ignored by `.gitignore` to prevent committing multi-megabyte/gigabyte binary weights to source control.

## Inspected Custom Checkpoint Details

- **Filename**: `best new.pt`
- **Architecture**: Ultralytics YOLO11n (`yolo11n.pt` base, 2,605,369 parameters)
- **Supported Classes**:
  - `0`: `crossing`
  - `1`: `track-signal-front`
  - `2`: `track-sign-front`
- **Training Source**: Trained in Google Colab (`/content/T225/railway_detector`, 20 epochs, mAP50 = 0.316)
- **Application Role**: Infrastructure and signage detection at railway level crossings.
