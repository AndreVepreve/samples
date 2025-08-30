# ML Starter Pack (Core Models)

This folder mirrors the chapter you asked for and ships with:
- `introduction_to_machine_learning.md` — full printable chapter
- `core_model_zoo_handout.md` — compact cheat sheets for each model
- `notebooks/` — runnable templates (tuning + evaluation)
- `data/` — place CSVs or Parquet here (kept empty)
- `models/` — saved pipelines via Joblib

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import sklearn, imblearn; print(sklearn.__version__)"
```

Then open the notebooks in `notebooks/` and run end-to-end.

## Notes
- Pipelines keep preprocessing inside CV and at serve time to avoid leakage.
- KMeans uses `n_init='auto'` (works on scikit-learn ≥1.4). 
- For probability-sensitive use-cases, calibrate and plot reliability.
