# Copilot Instructions for MLproject1

## Project Overview
This is an end-to-end machine learning project focused on student performance prediction. The codebase is organized for modularity, reproducibility, and experiment tracking.

## Architecture & Key Components
- **src/components/**: Core ML pipeline modules
  - `data_ingestion.py`: Loads and splits raw data
  - `data_transformation.py`: Preprocesses features, encodes, scales
  - `model_trainer.py`: Trains models, saves artifacts
- **src/pipeline/**: Orchestrates workflows
  - `train_pipeline.py`: End-to-end training pipeline
  - `predict_pipeline.py`: Inference pipeline
- **notebook/**: Jupyter notebooks for EDA and experimentation
- **logs/**: Stores log files for runs and errors
- **mlproject.egg-info/**: Packaging metadata (auto-generated)

## Data Flow
- Raw data (`notebook/data/stud.csv`) → Ingestion → Transformation → Model Training → Evaluation/Prediction
- Artifacts and logs are saved for reproducibility.

## Developer Workflows
- **Install dependencies:**
  ```powershell
  pip install -r requirements.txt
  ```
- **Run training pipeline:**
  ```powershell
  python src/pipeline/train_pipeline.py
  ```
- **Run prediction pipeline:**
  ```powershell
  python src/pipeline/predict_pipeline.py
  ```
- **Jupyter notebooks:**
  Use for EDA and quick prototyping. Ensure required packages are installed in the notebook kernel.

## Conventions & Patterns
- Custom exceptions are defined in `src/exception.py`.
- Logging is handled via `src/logger.py` and outputs to `logs/`.
- All core logic is modularized under `src/components/`.
- Pipelines are orchestrated in `src/pipeline/` for separation of concerns.
- Use relative imports within `src/`.

## Integration Points
- External dependencies: numpy, pandas, seaborn, catboost, scikit-learn, etc. (see `requirements.txt`)
- Data files: `notebook/data/stud.csv`
- Model artifacts and logs: `logs/`, `catboost_info/`

## Examples
- To add a new model, extend `model_trainer.py` and update `train_pipeline.py`.
- For new data sources, update `data_ingestion.py` and ensure transformation logic in `data_transformation.py`.

---
For questions or unclear conventions, review the relevant module in `src/` or ask for clarification.
