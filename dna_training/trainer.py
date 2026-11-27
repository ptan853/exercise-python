"""Explicit fresh experiments and epoch-boundary resume; no training on import."""
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math
import os
import random
import uuid
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .model import ModelConfig, DNATransformer
from .data import DNADataset, DNACollator, build_dataloaders

@dataclass
class TrainConfig:
    max_epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-4
    encoder_learning_rate: float | None = None
    weight_decay: float = .01
    classification_weight: float = 1.0
    selection_metric: str = 'val_cls_loss'
    mlm_weight: float = 1.0
    mlm_probability: float = .15
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = 'auto'
    output_dir: str = 'runs'
    experiment_name: str = 'baseline'
    patience: int = 8
    min_delta: float = .001
    scheduler_patience: int = 3
    scheduler_factor: float = .5
    min_lr: float = 1e-6
    log_every_steps: int = 20
    train_clean_eval: bool = True
    class_weighting: str = 'none'
    weight_cap: float = 5.0
    sampling: str = 'shuffle'
    sampling_power: float = 1.0
    epoch_samples: int | None = None
    freeze_encoder: bool = False

    def validate(self):
        if self.encoder_learning_rate is not None and (not math.isfinite(self.encoder_learning_rate) or self.encoder_learning_rate <= 0):
            raise ValueError("encoder_learning_rate must be finite and positive")
        if self.freeze_encoder and (self.mlm_weight != 0 or self.classification_weight <= 0):
            raise ValueError('Frozen encoder probes require pure classification')
        if self.classification_weight < 0 or self.mlm_weight < 0 or not math.isfinite(self.classification_weight+self.mlm_weight):
            raise ValueError('Objective weights must be finite and nonnegative')
        if self.classification_weight+self.mlm_weight == 0:
            raise ValueError('At least one objective must be enabled')
        if self.selection_metric not in {'val_cls_loss','val_mlm_loss'}:
            raise ValueError('Unknown selection metric')
        if self.selection_metric=='val_cls_loss' and self.classification_weight==0:
            raise ValueError('Pure MLM must select by val_mlm_loss')
        if self.selection_metric=='val_mlm_loss' and (self.mlm_weight==0 or self.mlm_probability==0):
            raise ValueError('MLM selection requires MLM and positive masking probability')
        if self.class_weighting not in {'none','inverse_sqrt','inverse'}:
            raise ValueError('Unknown class_weighting')
        if self.sampling not in {'shuffle','weighted'}:
            raise ValueError('Unknown sampling strategy')
        if self.weight_cap < 1 or self.sampling_power < 0:
            raise ValueError('Invalid class weighting/sampling parameters')
        if self.epoch_samples is not None and (self.epoch_samples < 1 or self.sampling != 'weighted'):
            raise ValueError('epoch_samples requires weighted sampling and a positive value')
        if self.max_epochs < 1 or self.batch_size < 1 or self.log_every_steps < 1:
            raise ValueError('Epochs, batch size and log interval must be positive')
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.mlm_weight < 0 or self.gradient_clip <= 0:
            raise ValueError('Invalid optimization parameters')
        if not 0 <= self.mlm_probability <= 1 or not 0 < self.scheduler_factor < 1:
            raise ValueError('Invalid MLM/scheduler parameters')
        if self.patience < 0 or self.scheduler_patience < 0 or self.min_delta < 0 or self.min_lr < 0:
            raise ValueError('Invalid early-stop/scheduler parameters')
        if Path(self.experiment_name).name != self.experiment_name or not self.experiment_name:
            raise ValueError('experiment_name must be a plain directory name')

@dataclass
class TrainingResult:
    model: DNATransformer
    run_dir: Path
    history: list
    best_checkpoint: Path
    last_checkpoint: Path


def select_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available')
    return device


def tokenizer_state(tokenizer):
    return {'token_to_id': dict(tokenizer.token_to_id),
            'merges': [list(p) for p in tokenizer.merges],
            'base_tokens': list(tokenizer.base_tokens),
            'special_tokens': list(tokenizer.special_tokens)}


def _fingerprint(train, validation):
    # Order is included because it affects seeded sampling and fixed MLM masks.
    digest = hashlib.sha256()
    for name, records in [('train', train), ('validation', validation)]:
        digest.update(name.encode())
        for r in records:
            digest.update(json.dumps([r['species'],r['sequence']],ensure_ascii=False).encode())
            digest.update(b'\n')
    return digest.hexdigest()


def _metrics(matrix):
    c = matrix.double()
    tp = c.diag()
    denom = c.sum(0) + c.sum(1)
    f1 = 2 * tp / denom.clamp_min(1)
    return {'accuracy': (tp.sum() / c.sum().clamp_min(1)).item(),
            'macro_f1': f1.mean().item()}


def _losses(outputs, batch, weight, class_weights=None, classification_weight=1.0):
    cls = F.cross_entropy(outputs['class_logits'], batch['labels'])
    count = int(batch['mlm_labels'].ne(-100).sum().item())
    mlm = (F.cross_entropy(outputs['mlm_logits'].flatten(0,1),
                           batch['mlm_labels'].flatten(),ignore_index=-100)
           if count else outputs['mlm_logits'].sum() * 0.)
    objective_cls = (F.cross_entropy(outputs['class_logits'],batch['labels'],weight=class_weights)
                     if class_weights is not None else cls)
    # Disabled heads have no gradient, so AdamW also skips their weight decay.
    total = classification_weight*objective_cls if classification_weight else weight*mlm
    if classification_weight and weight: total = total + weight*mlm
    return total, cls, mlm, count, objective_cls


def _epoch(model, loader, device, cfg, optimizer=None, writer=None, global_step=0,
           class_weights=None):
    training = optimizer is not None
    model.train(training)
    total_n = targets_n = 0
    cls_sum = mlm_sum = objective_sum = objective_denominator = 0.
    matrix = torch.zeros(model.num_classes,
                         model.num_classes,dtype=torch.long)
    for index, cpu in enumerate(loader, 1):
        batch = {k:v.to(device) for k,v in cpu.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(batch['input_ids'],batch['attention_mask'])
            total, cls, mlm, n_targets, objective_cls = _losses(
                output,batch,cfg.mlm_weight,class_weights if training else None,cfg.classification_weight)
            if not torch.isfinite(total).item():
                raise RuntimeError('Non-finite loss')
            if training:
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip,
                                                       error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
        n = batch['labels'].numel()
        total_n += n
        cls_sum += cls.item() * n
        mass = class_weights[batch['labels']].sum().item() if training and class_weights is not None else n
        objective_sum += objective_cls.item() * mass
        objective_denominator += mass
        mlm_sum += mlm.item() * n_targets
        targets_n += n_targets
        truth = batch['labels'].detach().cpu()
        pred = output['class_logits'].detach().argmax(-1).cpu()
        classes = matrix.shape[0]
        matrix += torch.bincount(truth*classes+pred,minlength=classes**2).reshape(classes,classes)
        if training and (index == 1 or global_step % cfg.log_every_steps == 0):
            for tag,value in [('cls_loss',cls.item()),('mlm_loss',mlm.item()),
                              ('total_loss',total.item()),('objective_cls_loss',objective_cls.item()),('gradient_norm',norm.item()),
                              ('learning_rate',optimizer.param_groups[0]['lr'])]:
                writer.add_scalar('step/train_'+tag,value,global_step)
            writer.flush()
            print(f'  batch {index}/{len(loader)} cls={cls.item():.4f} mlm={mlm.item():.4f}',flush=True)
    if not total_n:
        raise ValueError('Empty loader')
    cls_mean,mlm_mean = cls_sum/total_n,mlm_sum/max(targets_n,1)
    objective_mean=objective_sum/objective_denominator
    return {'cls_loss':cls_mean,'objective_cls_loss':objective_mean,'mlm_loss':mlm_mean,
            'total_loss':cfg.classification_weight*objective_mean+cfg.mlm_weight*mlm_mean,**_metrics(matrix)},global_step


def _atomic_save(state, path):
    temp = path.with_suffix('.tmp')
    torch.save(state,temp)
    os.replace(temp,path)


def _write_history(path,history):
    if not history:return
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[-1]))
        w.writeheader();w.writerows(history)
    os.replace(temp,path)


def load_model(checkpoint_path, device='cpu'):
    """Load NEW-format checkpoints for inference. Only load files you trust."""
    state = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('Use init_from for legacy notebook checkpoints')
    model = DNATransformer(len(state['tokenizer']['token_to_id']),len(state['label_to_id']),
                           state['tokenizer']['token_to_id']['[PAD]'],ModelConfig(**state['model_config']))
    model.load_state_dict(state['model_state_dict'])
    model.to(select_device(device)).eval()
    return model,state


def train_experiment(train_records, val_records, tokenizer, label_to_id,
                     model_config=None, train_config=None, *, resume_from=None, init_from=None, experiment_context=None):
    """resume_from continues last.pt; init_from starts a NEW run from saved weights.

    Do not pass test records. Checkpoints are saved only after completed epochs.
    Same-device CPU resume is tested; bitwise GPU/cross-device reproducibility is
    not guaranteed. num_workers=0 is intentional for exact collator-state restore.
    """
    mc,tc = model_config or ModelConfig(),train_config or TrainConfig()
    mc.validate();tc.validate()
    if resume_from and init_from:
        raise ValueError('resume_from and init_from are mutually exclusive')
    if sorted(label_to_id.values()) != list(range(len(label_to_id))):
        raise ValueError('Labels must be contiguous IDs starting at zero')
    tok = tokenizer_state(tokenizer)
    if sorted(tok['token_to_id'].values()) != list(range(len(tok['token_to_id']))):
        raise ValueError('Token IDs must be contiguous')
    for name in ['[PAD]','[UNK]','[CLS]','[MASK]']:
        if name not in tok['token_to_id']:
            raise ValueError(f'Missing {name}')
    if not train_records or not val_records:
        raise ValueError('Training and validation records must be nonempty')
    if {r['species'] for r in train_records} != set(label_to_id):
        raise ValueError('Training classes do not match label map')
    normalize = getattr(tokenizer,'normalize',lambda s:s.upper())
    a={normalize(r['sequence']) for r in train_records}
    b={normalize(r['sequence']) for r in val_records}
    if a & b:
        raise ValueError('Identical normalized sequence crosses train/validation split')
    fingerprint=_fingerprint(train_records,val_records)
    saved=None
    if resume_from or init_from:
        saved=torch.load(resume_from or init_from,map_location='cpu',weights_only=True)
        oldtok=saved['tokenizer'].copy()
        oldtok['merges']=[list(p) for p in oldtok['merges']]
        if oldtok != tok or saved['label_to_id'] != label_to_id:
            raise ValueError('Checkpoint tokenizer/label mapping mismatch')
        if init_from and saved.get('format_version') == 1:
            previous=asdict(ModelConfig(**saved['model_config']))
            requested=asdict(mc)
            # Pooling adds no parameters; transfer is valid across CLS/mean.
            for key in ['pooling','classifier_hidden_dim','classifier_dropout']:
                previous.pop(key);requested.pop(key)
            if previous!=requested:
                raise ValueError('Weight initialization requires matching model configuration')
        if init_from and saved.get('format_version') != 1 and mc != ModelConfig():
            raise ValueError('Legacy checkpoint has no structured architecture config; only original default architecture is supported')
    if resume_from:
        if saved.get('format_version') != 1:
            raise ValueError('Legacy checkpoints lack resume state; use init_from for weight initialization')
        if Path(resume_from).name != 'last.pt':
            raise ValueError('Resume from last.pt; use init_from to branch from best.pt')
        if asdict(ModelConfig(**saved['model_config'])) != asdict(mc) or saved['data_fingerprint'] != fingerprint:
            raise ValueError('Model configuration or data split differs from checkpoint')
        mutable={'max_epochs','device','output_dir','experiment_name','log_every_steps','patience'}
        for key,value in asdict(tc).items():
            if key not in mutable and asdict(TrainConfig(**saved['train_config']))[key] != value:
                raise ValueError(f'Resume configuration changed: {key}; start a new experiment')
        if tc.max_epochs <= saved['epoch']:
            raise ValueError('max_epochs must exceed the saved epoch')
    device=select_device(tc.device)
    random.seed(tc.seed);torch.manual_seed(tc.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(tc.seed)
    if device.type=='mps':torch.mps.manual_seed(tc.seed)
    model=DNATransformer(len(tok['token_to_id']),len(label_to_id),tok['token_to_id']['[PAD]'],mc).to(device)
    model.freeze_encoder=tc.freeze_encoder
    if tc.freeze_encoder:
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('classification_head.'))
    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters=[p for p in model.parameters() if p.requires_grad]
    if tc.encoder_learning_rate is not None and not tc.freeze_encoder:
        parameters=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('classification_head.')],'lr':tc.learning_rate},
                    {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('classification_head.')],'lr':tc.encoder_learning_rate}]
    optimizer=torch.optim.AdamW(parameters,lr=tc.learning_rate,weight_decay=tc.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=tc.scheduler_factor,
                    patience=tc.scheduler_patience,min_lr=tc.min_lr)
    loaders=build_dataloaders(train_records,val_records,tokenizer,label_to_id,mc,tc)
    train_loader,val_loader=loaders['train'],loaders['validation']
    val_mlm_loader,train_clean_loader=loaders['validation_mlm'],loaders['train_clean']
    train_collator,val_collator=loaders['train_collator'],loaders['val_collator']
    shuffle=loaders['shuffle_generator']
    weights=loaders['class_weights'].to(device) if tc.class_weighting!='none' else None
    start,step,best,stale=1,0,float('inf'),0
    early_stop_reference=float('inf')
    history=[]
    reset_classifier=False
    if saved:
        reset_classifier=bool(init_from and ModelConfig(**saved.get('model_config',{})).classifier_hidden_dim != mc.classifier_hidden_dim)
        if reset_classifier:
            state={k:v for k,v in saved['model_state_dict'].items() if not k.startswith('classification_head.')}
            missing=model.load_state_dict(state,strict=False)
            if missing.unexpected_keys or any(not k.startswith('classification_head.') for k in missing.missing_keys):
                raise ValueError('Unexpected backbone transfer mismatch')
        else:
            model.load_state_dict(saved['model_state_dict'])
    if resume_from:
        run_dir=Path(resume_from).resolve().parent.parent
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start,step=saved['epoch']+1,saved['global_step']
        best,stale=saved.get('best_selection_loss',saved['best_val_cls_loss']),saved['early_stop_count']
        early_stop_reference=saved['early_stop_reference']
        history=saved['history']
        train_collator.generator.set_state(saved['collator_rng_state'])
        shuffle.set_state(saved['shuffle_rng_state'])
        random.setstate(saved['python_rng_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type=='cuda' and saved.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        if device.type=='mps' and saved.get('mps_rng_state') is not None:
            torch.mps.set_rng_state(saved['mps_rng_state'])
        # last.pt is the committed state. CSV is a derived, repairable report.
        # The resolved supplied path is the run's current last.pt (not an old alias).
        _write_history(run_dir/'metrics.csv',history)
        if history and history[-1][tc.selection_metric] == best:
            # Recover a save interrupted after committing last.pt but before best.pt.
            _atomic_save(saved,run_dir/'checkpoints/best.pt')
    else:
        run_dir=(Path(tc.output_dir)/f'{tc.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}').resolve()
    checkpoints=run_dir/'checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
    if init_from:
        transferred=all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k].cpu())
                        for k,v in model.state_dict().items() if not k.startswith(('classification_head.','mlm_head.')))
        if not transferred:raise RuntimeError('Backbone transfer verification failed')
        (run_dir/'initialization.json').write_text(json.dumps({
            'checkpoint':str(Path(init_from).resolve()),'source_epoch':saved.get('epoch'),
            'source_training':saved.get('train_config'),
            'backbone_verified_before_training':transferred,
            'optimizer':'fresh','classification_head':'fresh' if reset_classifier else 'loaded',
            'source_classifier_trained':saved.get('train_config',{}).get('classification_weight',1)>0},indent=2))
    (run_dir/'config.json').write_text(json.dumps({'model':asdict(mc),'training':asdict(tc),
        'data_fingerprint':fingerprint,'label_to_id':label_to_id,'tokenizer':tok,
        'torch_version':str(torch.__version__),'device':str(device),
        'trainable_parameters':trainable_parameters,'pooling':mc.pooling,'freeze_encoder':tc.freeze_encoder,
        'resume_from':str(resume_from) if resume_from else None,
        'init_from':str(init_from) if init_from else None,
        'experiment_context':experiment_context},indent=2),encoding='utf-8')
    balance=loaders['balance']
    balance['label_to_id']=label_to_id
    (run_dir/'balance.json').write_text(json.dumps(balance,indent=2),encoding='utf-8')
    # Preserve the actual inputs/split independently of notebook state.
    if not resume_from:
        manifest={name:[{'species':r['species'],'sequence':r['sequence'],
                           'accession':r.get('accession'),'allele':r.get('allele'),
                           'region':r.get('region')} for r in records]
                  for name,records in [('train',train_records),('validation',val_records)]}
        (run_dir/'splits.json').write_text(json.dumps(manifest),encoding='utf-8')
    writer=SummaryWriter(str(run_dir/'tensorboard'),purge_step=step+1 if resume_from else None,flush_secs=10)
    print(f'Device: {device}; run: {run_dir}; epochs: {start}..{tc.max_epochs}',flush=True)
    try:
        for epoch in range(start,tc.max_epochs+1):
            print(f'Epoch {epoch}',flush=True)
            train,step=_epoch(model,train_loader,device,tc,optimizer,writer,step,class_weights=weights)
            val,_=_epoch(model,val_loader,device,tc)
            row={'epoch':epoch,**{'train_'+k:v for k,v in train.items()},
                 **{'val_'+k:val[k] for k in ['cls_loss','accuracy','macro_f1']}}
            if tc.train_clean_eval:
                clean,_=_epoch(model,train_clean_loader,device,tc)
                row.update({'train_clean_'+k:clean[k] for k in ['cls_loss','accuracy','macro_f1']})
            if tc.mlm_weight>0:
                val_collator.generator.manual_seed(tc.seed+1000)
                masked,_=_epoch(model,val_mlm_loader,device,tc)
                row.update({'val_masked_cls_loss':masked['cls_loss'],
                            'val_mlm_loss':masked['mlm_loss'],'val_joint_loss':masked['total_loss']})
            # Selection drives checkpointing, LR scheduler and early stopping consistently.
            selection_value=row[tc.selection_metric]
            improved=selection_value<best
            significant=selection_value<early_stop_reference-tc.min_delta
            stale=0 if significant else stale+1
            if significant:early_stop_reference=selection_value
            if improved:best=selection_value
            scheduler.step(selection_value)
            row['learning_rate']=optimizer.param_groups[0]['lr']
            history.append(row)
            for key,value in row.items():
                if key!='epoch':writer.add_scalar('epoch/'+key,value,epoch)
            state={'format_version':1,'epoch':epoch,'global_step':step,
                   'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                   'scheduler_state_dict':scheduler.state_dict(),
                   'best_val_cls_loss':min(r['val_cls_loss'] for r in history),
                   'best_selection_loss':best,'selection_metric':tc.selection_metric,'selection_value':selection_value,
                   'early_stop_count':stale,'early_stop_reference':early_stop_reference,'history':history,'model_config':asdict(mc),
                   'train_config':asdict(tc),'tokenizer':tok,'label_to_id':label_to_id,
                   'data_fingerprint':fingerprint,'balance':balance,'experiment_context':experiment_context,'torch_rng_state':torch.get_rng_state(),
                   'python_rng_state':random.getstate(),'shuffle_rng_state':shuffle.get_state(),
                   'collator_rng_state':train_collator.generator.get_state(),
                   'cuda_rng_states':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   'mps_rng_state':torch.mps.get_rng_state() if device.type=='mps' else None}
            _atomic_save(state,checkpoints/'last.pt')
            if improved:_atomic_save(state,checkpoints/'best.pt')
            _write_history(run_dir/'metrics.csv',history)
            writer.flush()
            print(f"  val cls={val['cls_loss']:.4f}; acc={val['accuracy']:.2%}; F1={val['macro_f1']:.4f}"+
                  f'; {tc.selection_metric}={selection_value:.4f}'+('; saved best' if improved else ''),flush=True)
            if tc.patience and stale>=tc.patience:
                print('Early stopping',flush=True);break
    except KeyboardInterrupt:
        print('Interrupted. last.pt is the last COMPLETED epoch; current partial epoch was not saved.',flush=True)
        raise
    finally:
        writer.close()
    return TrainingResult(model,run_dir,history,checkpoints/'best.pt',checkpoints/'last.pt')
