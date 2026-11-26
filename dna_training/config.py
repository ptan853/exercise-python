"""Serializable experiment settings. Defaults preserve the current split."""
from dataclasses import dataclass,field,asdict
from pathlib import Path
import json
from .model import ModelConfig
from .trainer import TrainConfig

@dataclass
class DataConfig:
    source_path: str = 'reference_nt.fasta'
    cache_dir: str = 'prepared'
    seed: int = 42
    val_fraction: float = .1
    test_fraction: float = .1
    min_class_records: int = 3
    strip_species_suffix: bool = True
    exclude_known_long_record: bool = True

    def validate(self):
        if not 0 < self.val_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError('Validation/test fractions must be in (0,1)')
        if self.val_fraction+self.test_fraction >= 1:
            raise ValueError('Validation + test fractions must be below 1')
        if self.min_class_records < 3:
            raise ValueError('min_class_records must be at least 3')

@dataclass
class BPEConfig:
    vocab_size: int = 128
    min_frequency: int = 2
    def validate(self):
        if self.vocab_size < 9 or self.min_frequency < 1:
            raise ValueError('BPE requires vocab_size>=9 and min_frequency>=1')

@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    def validate(self):
        for section in [self.data,self.bpe,self.model,self.training]:section.validate()
    def save(self,path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self),indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        allowed={'data','bpe','model','training'}
        if set(d)-allowed:raise ValueError('Unknown configuration section')
        obj=cls(DataConfig(**d.get('data',{})),BPEConfig(**d.get('bpe',{})),
                ModelConfig(**d.get('model',{})),TrainConfig(**d.get('training',{})))
        obj.validate();return obj
