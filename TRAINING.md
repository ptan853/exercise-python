# Local training modules

`dna_training/model.py` contains the existing Post-LN/ReLU Transformer and fixed
sinusoidal positions. `data.py` contains datasets and dynamic MLM. `trainer.py`
owns training/validation, TensorBoard, scheduling, early stopping and checkpoints.
Importing these modules never starts training. Preprocessing and BPE remain in
the notebook so the data decisions remain visible.

## Notebook workflow

1. Select Python (dna-exercises), restart the kernel, and execute Test 3 in order.
2. Prepare the existing splits and fit the tokenizer only on training data.
3. Edit ModelConfig and TrainConfig in the notebook. The trainer rebuilds loaders
   from config; preview loaders are only for inspecting batches.
4. Choose fresh, resume, or weights in the final cell, then execute it intentionally.
   No full experiment has been launched by this refactor.

Fresh always builds a new model and optimizer after setting its seed. Resume
requires a NEW-format last.pt and matching tokenizer, label map, ordered data
splits, architecture and training settings. max_epochs is the TOTAL target epoch,
not the number of additional epochs. Only max_epochs, patience, log interval and
execution device/location may change on resume. Device changes can alter results.
Do not run two trainers in the same experiment directory.

Weights mode loads only weights into a NEW experiment/optimizer/schedule. It can
load old notebook best.pt/last.pt when architecture and token/label mapping match.
It is deliberately not called complete resume: old files lack loader/MLM RNG and
scheduler state. A new-format resume preserves CPU/Python and available GPU RNG,
shuffle generator, dynamic masking generator, optimizer, scheduler, history and
monitor state. CPU epoch-boundary equivalence is tested; GPU bitwise equivalence
is not guaranteed.

## Monitoring and selection

Train metrics use masked inputs and dropout. Optional train_clean_eval evaluates
all training samples without masking/dropout once per epoch for a comparable
classification diagnostic. Validation classification always uses clean inputs.
Validation MLM uses repeatable masks. Disabling MLM (mlm_weight=0) also disables
training corruption and its auxiliary validation pass.

best.pt tracks minimum clean validation classification loss. ReduceLROnPlateau
uses this loss. Early stopping monitors improvements of at least min_delta over
its last significant improvement; patience=0 disables early stopping.

Losses are aggregated by samples (classification) or selected tokens (MLM).
Snapshots contain model/train configs, tokenizer and labels. splits.json records
training and validation inputs. The trainer does not consume the test set.
Keyboard interruption retains the last COMPLETED epoch; in-progress batches are
not saved and repeat after resume. Checkpoints should only be loaded from trusted
sources; torch.load uses weights_only=True and map_location='cpu'.

## Inference from a new checkpoint

```python
from dna_training import load_model
model, state = load_model('runs/EXPERIMENT/checkpoints/best.pt', device='auto')
```

Use state['tokenizer'] and state['label_to_id'] to reconstruct the same tokenization
and interpretation. Do not train a new vocabulary for an existing model.

## TensorBoard locally or remotely

```sh
conda run -n dna-exercises tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

For a server, run TensorBoard there and forward from your laptop:

```sh
ssh -N -L 16006:127.0.0.1:6006 USER@SERVER
```

Then open http://localhost:16006. No public TensorBoard listener is needed.
Step curves appear during training; epoch curves appear after validation finishes.
Use `python -u your_script.py` for immediate console output on a server.

## Checks

```sh
conda run -n dna-exercises python -m unittest discover -s tests -v
```

These use tiny synthetic sequences, never the real test set. Current data splitting
still only removes exact duplicates; gene/sequence similarity leakage and species
normalization remain methodological limitations. The modules do not silently
change those choices. The required final confusion matrix must have predicted
classes in rows; the trainer's internal metric accumulator uses true rows.
