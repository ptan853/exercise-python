# Local training modules

Goal: extract existing model and batching into importable modules; add configuration,
validation, portable checkpoints, explicit fresh/resume/weight initialization, and logs.

1. Add integration tests for model masking, no-MLM batches, one-epoch training,
   CPU checkpoint round trip, strict resume compatibility and continuation.
2. Keep original two-layer Post-LN/ReLU encoder in model.py; configurable dataclass.
3. Place Dataset and collator in data.py; tokenizer and split preparation stay in notebook.
4. Trainer owns model/optimizer/plateau scheduler, RNG state and metric aggregation.
   Validate and fingerprint inputs before creating a run. Save best/last atomically.
5. Replace only third-exercise Dataset/model/training cells with imports and config;
   leave preprocessing, BPE, exercises 1/2 and old runs unchanged.
6. Run integration tests, old checkpoint weight loading and real-data batching checks.
   Do not train a full real-data experiment or push changes.
