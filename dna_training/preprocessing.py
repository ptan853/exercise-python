"""FASTA audit and historical per-species stratified record split; no gene grouping."""
from collections import Counter,defaultdict
from pathlib import Path
from dataclasses import dataclass,asdict
import hashlib,json,random
from .tokenizer import DNABytePairEncoder

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def data_settings(config):
    return {k:v for k,v in asdict(config).items() if k not in {'source_path','cache_dir'}}

def read_fasta(path,strip_species_suffix=True):
    records=[];header=None;parts=[]
    def save():
        fields=header.split('|')
        if len(fields)<5:raise ValueError('FASTA header needs at least five IMGT fields')
        species=fields[2].strip()
        sequence=''.join(parts).upper()
        if not sequence:raise ValueError('Empty FASTA sequence')
        records.append({'accession':fields[0].strip(),'allele':fields[1].strip(),
            'species_raw':species,'species':species.split('_',1)[0] if strip_species_suffix else species,
            'region':fields[4].strip(),'sequence':sequence})
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:continue
            if line.startswith('>'):
                if header is not None:save()
                header=line[1:];parts=[]
            else:
                if header is None:raise ValueError('Sequence before FASTA header')
                parts.append(line)
    if header is not None:save()
    if not records:raise ValueError('Empty FASTA')
    return records

def split_records(records,config):
    config.validate();excluded=[];kept=[]
    for r in records:
        exclude=(config.exclude_known_long_record and r['accession']=='IMGT000237'
                 and r['allele']=='TRDC*01' and r['species_raw']=='Mesocricetus auratus_SY011'
                 and r['region']=='EX4UTR' and len(r['sequence'])==37613)
        (excluded if exclude else kept).append(r)
    groups=defaultdict(list)
    for r in kept:groups[r['sequence']].append(r)
    clean=[];conflicts=[]
    for group in groups.values():
        if len({r['species'] for r in group})>1:conflicts.append(group)
        else:clean.append(group[0])
    by_species=defaultdict(list)
    for r in clean:by_species[r['species']].append(r)
    rng=random.Random(config.seed)
    splits={'train':[],'validation':[],'test':[]};rare=[]
    for species in sorted(by_species):
        group=by_species[species].copy();rng.shuffle(group)
        if len(group)<config.min_class_records:rare.extend(group);continue
        nv=max(1,int(len(group)*config.val_fraction));nt=max(1,int(len(group)*config.test_fraction))
        if nv+nt>=len(group):raise ValueError('Fractions leave no training records for '+species)
        splits['validation'].extend(group[:nv]);splits['test'].extend(group[nv:nv+nt]);splits['train'].extend(group[nv+nt:])
    if any(not v for v in splits.values()):raise ValueError('One split is empty')
    labels={s:i for i,s in enumerate(sorted({r['species'] for r in splits['train']}))}
    for name in ['validation','test']:
        if {r['species'] for r in splits[name]}!=set(labels):raise ValueError('Class coverage differs')
    # Check actual model-visible input, but do not silently change historical dedup rules.
    visible={name:{DNABytePairEncoder.normalize(r['sequence']) for r in rows} for name,rows in splits.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if visible[a]&visible[b]:raise ValueError('Normalized identical sequences cross splits: '+a+'/'+b)
    audit={'source_records':len(records),'excluded_records':excluded,
           'unique_sequences':len(groups),'conflicting_sequence_groups':len(conflicts),
           'conflicting_records':sum(map(len,conflicts)),'clean_records':len(clean),
           'same_species_duplicates_removed':len(kept)-sum(map(len,conflicts))-len(clean),
           'rare_records':rare,'class_counts':{n:dict(Counter(r['species'] for r in rows)) for n,rows in splits.items()},
           'split_method':'per_species_stratified_random_records',
           'limitations':['Species suffix stripping does not harmonize subspecies.',
                          'Exact deduplication does not prevent homologous genes crossing splits.']}
    return splits,labels,audit

@dataclass
class PreparedData:
    path: Path
    split_id: str
    prepared_id: str
    train_records: list
    val_records: list
    test_records: list
    tokenizer: DNABytePairEncoder
    label_to_id: dict
    metadata: dict


def load_prepared(path):
    path=Path(path)
    meta=json.loads((path/'metadata.json').read_text())
    split_path=(path/Path(meta['split_relative_path'])).resolve()
    splits=json.loads(split_path.read_text())
    if digest(splits)!=meta['splits_digest']:raise ValueError('Cached splits were modified')
    tok=DNABytePairEncoder.load(path/'tokenizer.json')
    if digest(tok.to_state())!=meta['tokenizer_digest']:raise ValueError('Cached tokenizer was modified')
    return PreparedData(path.resolve(),meta['split_id'],meta['prepared_id'],splits['train'],
        splits['validation'],splits['test'],tok,meta['label_to_id'],meta)


def prepare_data(data_config=None,bpe_config=None):
    from .config import DataConfig,BPEConfig
    dc,bc=data_config or DataConfig(),bpe_config or BPEConfig()
    dc.validate();bc.validate()
    source=Path(dc.source_path)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    split_id=digest({'schema':1,'source_sha256':sha,'settings':data_settings(dc)})
    prepared_id=digest({'schema':1,'split_id':split_id,'bpe':asdict(bc)})
    root=Path(dc.cache_dir).resolve();split_dir=root/'splits'/split_id
    path=root/'datasets'/prepared_id
    if (path/'metadata.json').exists():return load_prepared(path)
    split_dir.mkdir(parents=True,exist_ok=True);path.mkdir(parents=True,exist_ok=True)
    if (split_dir/'audit.json').exists():
        splits=json.loads((split_dir/'splits.json').read_text());audit=json.loads((split_dir/'audit.json').read_text())
        if digest(splits)!=audit['splits_digest']:raise ValueError('Cached split was modified')
        labels=audit['label_to_id']
    else:
        splits,labels,audit=split_records(read_fasta(source,dc.strip_species_suffix),dc)
        audit.update({'splits_digest':digest(splits),'label_to_id':labels,'source_sha256':sha,'data_settings':data_settings(dc)})
        (split_dir/'splits.json').write_text(json.dumps(splits),encoding='utf-8')
        (split_dir/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    tok=DNABytePairEncoder().fit([r['sequence'] for r in splits['train']],
                               vocab_size=bc.vocab_size,min_frequency=bc.min_frequency)
    tok.save(path/'tokenizer.json')
    length_stats={}
    for name,rows in splits.items():
        lengths=sorted(len(tok.encode(r['sequence'])) for r in rows)
        length_stats[name]={'records':len(rows),'min':lengths[0],'max':lengths[-1],
            'median':lengths[len(lengths)//2],'total_tokens':sum(lengths)}
    relative=Path(os_path_rel(split_dir/'splits.json',path))
    meta={'schema_version':1,'split_id':split_id,'prepared_id':prepared_id,
          'source_path':str(source),'source_sha256':sha,'data_config':asdict(dc),'bpe_config':asdict(bc),
          'split_relative_path':str(relative),'splits_digest':digest(splits),
          'tokenizer_digest':digest(tok.to_state()),'label_to_id':labels,'audit':audit,'token_lengths':length_stats}
    (path/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return load_prepared(path)


def os_path_rel(path,start):
    import os
    return os.path.relpath(path,start)
