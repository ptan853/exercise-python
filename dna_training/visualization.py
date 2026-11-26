"""File-based charts for notebooks and headless servers; no pandas dependency."""
from pathlib import Path
import csv,json
import numpy as np

def _read_history(run_dir):
    with (Path(run_dir)/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No completed epochs yet')
    return rows

def _values(rows,key):
    return [float(r[key]) if r.get(key) not in (None,'') else np.nan for r in rows]

def plot_training_curves(run_dir,output_path=None):
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir);rows=_read_history(run_dir)
    output=Path(output_path) if output_path else run_dir/'figures/training_curves.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    x=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    def line(ax,key,label):
        if any(r.get(key) not in (None,'') for r in rows):
            ax.plot(x,_values(rows,key),label=label,lw=1.8)
    clean='train_clean_cls_loss' in rows[-1]
    line(axes[0,0],'train_clean_cls_loss' if clean else 'train_cls_loss',
         'Train clean' if clean else 'Train masked/dropout')
    line(axes[0,0],'val_cls_loss','Validation clean')
    axes[0,0].set_title('Unweighted classification loss')
    line(axes[0,1],'train_clean_accuracy' if clean else 'train_accuracy',
         'Train clean' if clean else 'Train masked/dropout')
    line(axes[0,1],'val_accuracy','Validation clean')
    axes[0,1].set_title('Classification accuracy')
    line(axes[1,0],'train_clean_macro_f1','Train clean')
    line(axes[1,0],'val_macro_f1','Validation clean')
    axes[1,0].set_title('Macro-F1')
    line(axes[1,1],'learning_rate','Learning rate')
    axes[1,1].set_title('Learning rate')
    cfg=json.loads((run_dir/'config.json').read_text())
    mlm_enabled=cfg.get('training',{}).get('mlm_weight',1)>0
    if mlm_enabled:
        line(axes[2,0],'train_mlm_loss','Train MLM')
        line(axes[2,0],'val_mlm_loss','Validation MLM (fixed masks)')
    else:
        axes[2,0].text(.5,.5,'MLM disabled',ha='center',va='center',transform=axes[2,0].transAxes)
    axes[2,0].set_title('MLM loss (selected tokens only)')
    line(axes[2,1],'train_total_loss','Train objective')
    line(axes[2,1],'val_joint_loss','Validation joint objective')
    axes[2,1].set_title('Joint objective (masking / weighting may differ)')
    for ax in axes.flat:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8)
    fig.suptitle(run_dir.name);fig.savefig(output,dpi=160);plt.close(fig)
    return output

def plot_confusion(matrix,class_names,output_path):
    """Input matrix is counts with PREDICTED rows, TRUE columns."""
    import matplotlib.pyplot as plt
    matrix=np.asarray(matrix)
    totals=matrix.sum(0,keepdims=True)
    normalized=np.divide(matrix,totals,out=np.zeros_like(matrix,dtype=float),where=totals!=0)
    size=max(7,min(18,len(class_names)*.35))
    fig,ax=plt.subplots(figsize=(size+2,size),layout='constrained')
    image=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(class_names)),class_names,rotation=90,fontsize=7)
    ax.set_yticks(range(len(class_names)),class_names,fontsize=7)
    ax.set_xlabel('True species');ax.set_ylabel('Predicted species')
    ax.set_title('Confusion matrix (normalized by true species)')
    fig.colorbar(image,ax=ax,label='Fraction of each true class')
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160);plt.close(fig)
    return output

def compare_experiments(run_dirs,output_path='comparison.png',require_same_data=True):
    import matplotlib.pyplot as plt
    if not run_dirs:raise ValueError('No runs supplied')
    dirs=[Path(p) for p in run_dirs]
    configs=[json.loads((p/'config.json').read_text()) for p in dirs]
    if require_same_data and len({c['data_fingerprint'] for c in configs})>1:
        raise ValueError('Runs use different train/validation data; not a controlled comparison')
    # Same label vocabulary and tokenization, not merely the same raw split.
    if require_same_data and len({json.dumps(c['tokenizer'],sort_keys=True) for c in configs})>1:
        raise ValueError('Runs use different tokenizers; set require_same_data=False to compare explicitly')
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');summary=[]
    for d in dirs:
        rows=_read_history(d);x=[int(r['epoch']) for r in rows]
        for ax,key in zip(axes,['val_cls_loss','val_accuracy','val_macro_f1']):
            ax.plot(x,_values(rows,key),label=d.name)
            ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        best=min(rows,key=lambda r:float(r['val_cls_loss']))
        summary.append({'run':str(d),'selection':'minimum val_cls_loss','epoch':best['epoch'],
            **{k:best[k] for k in ['val_cls_loss','val_accuracy','val_macro_f1']}})
    axes[-1].legend(fontsize=6);fig.savefig(output,dpi=160);plt.close(fig)
    with output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    return output
