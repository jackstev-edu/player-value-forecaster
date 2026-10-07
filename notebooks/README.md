# Notebooks

Both open in Colab (badges in the top-level README) and read the published Hugging Face dataset, so they need no team Drive.

| Notebook | What it shows | Runtime |
| --- | --- | --- |
| `01_dataset_eda.ipynb` | The 500 hand-collected rows against the 500 augmented ones: separation checks, collection quality, the target and its survivor effect, what moves with it, missing values, one player's value history | under a minute, nothing to install |
| `02_end_to_end.ipynb` | Clones this repo, cross-validates the from-scratch LightGBM against three baselines, runs the off-the-shelf Chronos-Bolt on the same rows, then loads the published model from the Hub to forecast a player you describe | about 5 minutes on CPU |

To test them before publishing, set `PVF_LOCAL_DATASET` to `data/processed/hf_dataset` and `PVF_LOCAL_MODEL` to `data/processed/hf_models/lgbm-core`.
