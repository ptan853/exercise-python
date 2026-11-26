"""High-level entrypoint connecting frozen data/tokenizer assets to training."""
from dataclasses import asdict
from pathlib import Path
import json
from .preprocessing import data_settings
from .trainer import train_experiment

def run_experiment(prepared,config,*,resume_from=None,init_from=None):
    config.validate()
    recorded=prepared.metadata['data_config']
    actual=data_settings(config.data)
    if actual!={k:v for k,v in recorded.items() if k not in {'source_path','cache_dir'}}:
        raise ValueError('Data settings differ from prepared asset; prepare a new dataset')
    if asdict(config.bpe)!=prepared.metadata['bpe_config']:
        raise ValueError('BPE settings differ from prepared asset; prepare a new tokenizer')
    context={'prepared_id':prepared.prepared_id,'split_id':prepared.split_id,
             'prepared_path':str(prepared.path),'source_sha256':prepared.metadata['source_sha256'],
             'experiment_config':asdict(config),'tokenizer_digest':prepared.metadata['tokenizer_digest']}
    result=train_experiment(prepared.train_records,prepared.val_records,prepared.tokenizer,
        prepared.label_to_id,config.model,config.training,resume_from=resume_from,
        init_from=init_from,experiment_context=context)
    # Written only to the experiment; original preparation cache is never changed.
    (result.run_dir/'prepared_data.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
    return result
