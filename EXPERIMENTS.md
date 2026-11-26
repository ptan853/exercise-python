# Unified experiment workflow

Preprocessing/BPE, loaders, balancing, model/training, evaluation and figures now
live in `dna_training`. Imports never launch work. JSON configurations need no new
parser dependency. Run from the repository directory with Python (dna-exercises).

## Prepare and reuse data

```python
from dna_training import ExperimentConfig, prepare_data, load_prepared
cfg = ExperimentConfig.load('configs/classification.json')
prepared = prepare_data(cfg.data, cfg.bpe)
print(prepared.path)
# Later, or after copying the prepared/ tree to another machine:
# prepared = load_prepared('prepared/datasets/ID')
```

Default rules reproduce historical results: remove the explicitly identified long
record, strip species suffix after underscore, group exact uppercase sequences,
exclude cross-label conflicts, keep one same-label duplicate, per-species seeded
80/10/10 split with at least one record per split, exclude classes with <3 unique
records. This does not harmonize subspecies or group homologous genes.
Normalization collisions across splits raise an error rather than silently alter
historical splits. Metadata records exclusions, conflicts, rare records, per-class
counts and post-BPE length statistics. Empty/oversized model inputs raise errors.
No data is silently truncated.

Cache layout:

```
prepared/splits/SPLIT_ID/{splits.json,audit.json}
prepared/datasets/PREPARED_ID/{metadata.json,tokenizer.json}
```

Changing BPE settings reuses the split but learns a new training-only vocabulary.
Changing source content, split seed or cleaning settings creates a different split.
Data and vocabulary digests are checked on load. Copy the WHOLE prepared/ tree,
not just one dataset leaf: its metadata references the shared split relatively.
The readable source/cache paths are not identity keys, so relocation is supported.
Algorithm/schema changes require increasing the cache schema version in code.

BPE retains the notebook's deterministic adjacent-pair merge rules; special tokens
are PAD/UNK/CLS/MASK. N is a boundary for merges; other ambiguous bases become N.
The implementation omits merge candidates whose concatenated token is already in
the vocabulary, preserving prior experiments. It is an educational BPE variant,
not a claim of byte-for-byte equivalence to external tokenizer libraries.

## Controlled experiments

```python
from dataclasses import replace
from dna_training import run_experiment
cfg.training = replace(cfg.training, experiment_name='weighted',
    mlm_weight=0.0, class_weighting='inverse_sqrt', weight_cap=5.0,
    sampling='shuffle')
result = run_experiment(prepared, cfg)
```

Class weighting: `none`, `inverse_sqrt`, `inverse`. Compute inverse count powers
from TRAIN counts, divide by arithmetic mean, then cap at weight_cap (>=1).
Weighted cross entropy divides by the sum of selected sample class weights, as
PyTorch does. `train_cls_loss` remains ordinary unweighted CE; the new
`train_objective_cls_loss` is weighted CE. `train_total_loss` uses objective CE plus
weighted MLM. Across an epoch objective CE is aggregated by total weight mass,
ordinary CE by samples, and MLM by selected token count. This is not a plain
average of batch losses. Gradients still optimize each batch's weighted mean.

Sampling: `shuffle` or `weighted`. A record in class c has sampling weight
`count[c] ** (-sampling_power)`; weighted sampling uses replacement, with
`epoch_samples` draws (default: training-set size). power=1 balances EXPECTED
class frequency, not the class counts of every batch. It repeats existing data;
it does not create independent biological samples. Train metrics under sampling
refer to actual sampled draws; train_clean metrics use the full unique train set.
Validation/test always preserve their original distribution and unweighted loss.

Both switches can be enabled independently. Using both compounds minority emphasis;
prefer separate runs first. Each run writes `balance.json` with training counts,
weights, sampler settings and expected class probabilities. Tiny classes can still
be unlearnable; weighting does not solve insufficient or ambiguous information.

Example configurations:
- classification.json: no MLM, no weighting, ordinary shuffle.
- weighted.json: classification only, inverse-square-root weights.
- resampled.json: classification only, balanced expected sampling.
- joint-mlm.json: historical joint task with no balancing.

## Training and resume

```python
result = run_experiment(prepared, cfg)  # fresh
# cfg.training.max_epochs = 100  # TOTAL target, not additional epochs
# result = run_experiment(prepared, cfg, resume_from='runs/RUN/checkpoints/last.pt')
# result = run_experiment(prepared, cfg, init_from='runs/OLD/checkpoints/best.pt')
```

The low-level train_experiment API remains available. Old format-version-1 runs
can resume with default disabled balance settings. Changing weighting, sampling,
BPE, model or data during resume is rejected; create a fresh or weights-initialized
run instead. Checkpoint history is authoritative; an interrupted CSV export is repaired on resume.
Epoch-boundary CPU resume with Dropout, weighting and weighted
sampling is tested. Cross-device/GPU bitwise identity is not guaranteed.
Checkpoint and RNG behavior is documented in TRAINING.md. Existing run artifacts
are never migrated or rewritten by the refactor.

Run outputs include config.json, balance.json, splits.json, metrics.csv,
prepared_data.json, tensorboard/ and checkpoints/{best.pt,last.pt}. High-level
checkpoint context includes the full nested configuration and asset fingerprints.
`prepared_data.json` is finalized when run_experiment returns; the context is
already in config/checkpoints while training is running.

## Evaluation and plots

```python
from dna_training import evaluate_checkpoint, plot_training_curves, compare_experiments
report = evaluate_checkpoint(prepared, result.best_checkpoint, split='validation')
print(report.metrics)
print(report.per_class)
print(report.errors[:10])
plot_training_curves(result.run_dir)
compare_experiments(['runs/A','runs/B'], 'comparisons/a-vs-b.png')
```

Evaluation loads clean inputs and the saved model/config. It checks the tokenizer,
labels and train/validation fingerprint before inference. `split='test'` must be
explicitly requested after model selection; training never evaluates test data.
Outputs under evaluation/validation-best/ (or test-best/) are metrics.json,
classification_report.csv, errors.csv, predictions.csv, confusion_counts.csv and
confusion_matrix.png. Confusion counts have PREDICTED rows and TRUE columns; the
figure normalizes columns by true species. These validation reports are diagnostic,
not the final test-set assignment result.

Figures are saved, not automatically displayed, so headless servers work. In a
notebook use `display(Image(filename=str(path)))` from IPython.display. Comparison
plots include a CSV selected consistently by minimum validation CE, not by cherry-
picking a different epoch for each metric. Different data/tokenizer comparisons
are rejected by default; explicitly disable require_same_data only when intended.

## Server entrypoints and live logs

```sh
python -m dna_training --config configs/weighted.json --action prepare
python -u -m dna_training --config configs/weighted.json --action train
python -u -m dna_training --config configs/weighted.json --mode resume --checkpoint runs/RUN/checkpoints/last.pt
python -m dna_training --config configs/weighted.json --action evaluate --checkpoint runs/RUN/checkpoints/best.pt --split validation
python -m dna_training --config configs/weighted.json --action plot --run-dir runs/RUN
```

TensorBoard: `tensorboard --logdir runs --host 127.0.0.1 --port 6006`.
Forward from laptop with `ssh -N -L 16006:127.0.0.1:6006 USER@SERVER`, then open
http://localhost:16006. Enable Reload data in TensorBoard. Step curves update
throughout training, epoch curves only after full validation.

## Tests

`python -m unittest discover -s tests -v`

Synthetic tests cover cache/tokenizer round trips and tampering, balancing math,
MLM/PAD handling, numerical weighted CE, fresh/resume and sampling state recovery,
report matrix orientation and headless plot export. The real test split is never
used to select a model in these checks.
