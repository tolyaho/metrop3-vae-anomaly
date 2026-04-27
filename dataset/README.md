# Dataset Directory

This project expects the UCI MetroPT3 files locally under `dataset/raw/`:

```text
dataset/raw/MetroPT3(AirCompressor).csv
dataset/raw/Data Description_Metro.pdf
```

Raw data, preprocessed splits, and processed window arrays are intentionally not
tracked in Git because they are large generated/local artifacts.

Recreate derived data with:

```bash
python scripts/preprocess.py --config configs/data/archive_like_window60.json
python scripts/build_windows.py --config configs/features/window60_noscale.json
```
