# Superseded experiments

Kept for provenance; not used by the scanner or the current workflow (see `docs/research.md`).

- `finetune.py`: first CPU fine-tune (frozen probe vs. fine-tuned encoder; tripled document false positives).
- `ft_data.py`, `finetune2.py`: CPU fine-tune on a synthetic + real dataset with augmentation; replaced
  by the GPU notebook (`notebooks/build_notebook.py`) trained on public data only.
- `install_ft_artifacts.py`: installs a single fine-tuned model in place of the base probe; replaced by
  `research/accept_models.py` (acceptance tests) and `research/install_ensemble.py` (ensemble member).

These scripts still refer to the old repository layout and to local documents that are not included.
