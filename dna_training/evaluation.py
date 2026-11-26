"""Explicit validation/test evaluation, labels restored from checkpoint."""
from dataclasses import dataclass
from pathlib import Path
import csv,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix
from .trainer import load_model,tokenizer_state,_fingerprint
from .data import DNADataset,DNACollator
from .visualization import plot_confusion

@dataclass
class EvaluationResult:
    metrics: dict
    per_class: list
    errors: list
    confusion_pred_true: np.ndarray
    y_true: list
    y_pred: list
    output_dir: Path


def _csv(path,rows,fieldnames):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)


def evaluate_checkpoint(prepared,checkpoint_path,*,split='validation',device='auto',batch_size=32,output_dir=None):
    if split not in {'validation','test'}:raise ValueError('split must be validation or test')
    if batch_size<1:raise ValueError('batch_size must be positive')
    model,state=load_model(checkpoint_path,device=device)
    savedtok=state['tokenizer'].copy();savedtok['merges']=[list(p) for p in savedtok['merges']]
    if tokenizer_state(prepared.tokenizer)!=savedtok or prepared.label_to_id!=state['label_to_id']:
        raise ValueError('Prepared tokenizer/labels differ from checkpoint')
    if _fingerprint(prepared.train_records,prepared.val_records)!=state['data_fingerprint']:
        raise ValueError('Prepared train/validation split differs from checkpoint')
    records=prepared.val_records if split=='validation' else prepared.test_records
    dataset=DNADataset(records,prepared.tokenizer,prepared.label_to_id,state['model_config']['max_length'])
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,
                      collate_fn=DNACollator(prepared.tokenizer),num_workers=0)
    device=next(model.parameters()).device;y_true=[];y_pred=[];loss_sum=0.
    with torch.inference_mode():
        for batch in loader:
            logits=model(batch['input_ids'].to(device),batch['attention_mask'].to(device))['class_logits']
            loss_sum+=torch.nn.functional.cross_entropy(logits,batch['labels'].to(device),reduction='sum').item()
            y_true.extend(batch['labels'].tolist());y_pred.extend(logits.argmax(-1).cpu().tolist())
    names=[k for k,v in sorted(prepared.label_to_id.items(),key=lambda p:p[1])];ids=list(range(len(names)))
    report=classification_report(y_true,y_pred,labels=ids,target_names=names,output_dict=True,zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=ids)
    per_class=[{'species':name,**report[name]} for name in names]
    errors=[]
    for true in ids:
        for pred in ids:
            if true!=pred and cm[true,pred]:
                errors.append({'true_species':names[true],'predicted_species':names[pred],
                    'count':int(cm[true,pred]),'fraction_of_true_class':float(cm[true,pred]/cm[true].sum())})
    errors.sort(key=lambda r:-r['count'])
    metrics={'split':split,'checkpoint':str(Path(checkpoint_path).resolve()),'epoch':state['epoch'],
        'samples':len(y_true),'cls_loss':loss_sum/len(y_true),
        'accuracy':float(np.mean(np.array(y_true)==np.array(y_pred))),
        'macro_f1':report['macro avg']['f1-score'],'weighted_f1':report['weighted avg']['f1-score'],
        'zero_recall_classes':sum(r['recall']==0 for r in per_class),
        'confusion_orientation':'rows=predicted, columns=true',
        'prepared_id':prepared.prepared_id}
    checkpoint_path=Path(checkpoint_path)
    out=Path(output_dir) if output_dir else checkpoint_path.parent.parent/'evaluation'/f'{split}-{checkpoint_path.stem}'
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    _csv(out/'classification_report.csv',per_class,['species','precision','recall','f1-score','support'])
    _csv(out/'errors.csv',errors,['true_species','predicted_species','count','fraction_of_true_class'])
    predictions=[{'accession':r.get('accession',''),'allele':r.get('allele',''),'region':r.get('region',''),
        'true_species':names[t],'predicted_species':names[p],'correct':t==p,'base_length':len(r['sequence'])}
        for r,t,p in zip(records,y_true,y_pred)]
    _csv(out/'predictions.csv',predictions,['accession','allele','region','true_species','predicted_species','correct','base_length'])
    with (out/'confusion_counts.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['predicted_species / true_species']+names)
        for name,row in zip(names,cm.T.tolist()):w.writerow([name]+row)
    plot_confusion(cm.T,names,out/'confusion_matrix.png')
    return EvaluationResult(metrics,per_class,errors,cm.T,y_true,y_pred,out)
