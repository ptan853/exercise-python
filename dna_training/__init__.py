"""Configurable, reproducible DNA experiments; no work starts on import."""
from .model import ModelConfig, DNATransformer
from .trainer import TrainConfig, TrainingResult, train_experiment, load_model
from .config import DataConfig, BPEConfig, ExperimentConfig
from .tokenizer import DNABytePairEncoder
from .preprocessing import PreparedData, prepare_data, load_prepared
from .data import DNADataset, DNACollator, build_dataloaders
from .experiment import run_experiment
from .evaluation import EvaluationResult, evaluate_checkpoint
from .visualization import plot_training_curves, plot_confusion, compare_experiments
__all__ = [name for name in globals() if not name.startswith('_')]
