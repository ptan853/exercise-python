"""Integer encoding, dynamic batch padding and dynamic MLM corruption."""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DNADataset(Dataset):
    def __init__(self, records, tokenizer, label_to_id, max_length=512):
        self.samples = []
        cls_id = tokenizer.token_to_id['[CLS]']
        for record in records:
            if record['species'] not in label_to_id:
                raise ValueError(f"Unknown species: {record['species']}")
            ids = tokenizer.encode(record['sequence'])
            if not ids:
                raise ValueError('Empty sequence')
            ids = [cls_id] + ids
            if len(ids) > max_length:
                raise ValueError(f"Sequence {record.get('allele', '')} has {len(ids)} tokens; limit={max_length}")
            self.samples.append({'input_ids': torch.tensor(ids, dtype=torch.long),
                                 'labels': torch.tensor(label_to_id[record['species']], dtype=torch.long)})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]

class DNACollator:
    def __init__(self, tokenizer, use_mlm=False, mlm_probability=0.15, seed=42):
        if not 0 <= mlm_probability <= 1:
            raise ValueError('mlm_probability must be in [0, 1]')
        self.use_mlm = use_mlm
        self.mlm_probability = mlm_probability
        self.generator = torch.Generator().manual_seed(seed)
        self.pad_id = tokenizer.token_to_id['[PAD]']
        self.cls_id = tokenizer.token_to_id['[CLS]']
        self.mask_id = tokenizer.token_to_id['[MASK]']
        specials = set(tokenizer.special_tokens)
        self.ordinary_ids = torch.tensor([v for k,v in tokenizer.token_to_id.items()
                                         if k not in specials], dtype=torch.long)
        self.special_ids = [tokenizer.token_to_id[k] for k in specials]
        if not len(self.ordinary_ids):
            raise ValueError('Tokenizer has no ordinary tokens')
    def __call__(self, samples):
        ids = pad_sequence([s['input_ids'] for s in samples], batch_first=True,
                           padding_value=self.pad_id)
        mask = ids.ne(self.pad_id)
        targets = torch.full_like(ids, -100)
        if self.use_mlm:
            eligible = mask.clone()
            for value in self.special_ids:
                eligible &= ids.ne(value)
            selected = (torch.rand(ids.shape, generator=self.generator) < self.mlm_probability) & eligible
            targets[selected] = ids[selected]
            action = torch.rand(ids.shape, generator=self.generator)
            ids[selected & (action < .8)] = self.mask_id
            random_positions = selected & (action >= .8) & (action < .9)
            count = int(random_positions.sum())
            if count:
                choices = torch.randint(len(self.ordinary_ids), (count,), generator=self.generator)
                ids[random_positions] = self.ordinary_ids[choices]
        return {'input_ids': ids, 'attention_mask': mask, 'mlm_labels': targets,
                'labels': torch.stack([s['labels'] for s in samples])}


def build_dataloaders(train_records,val_records,tokenizer,label_to_id,model_config,train_config):
    """Central loader factory; no test set is sampled by the trainer.

    num_workers=0 preserves exact epoch-boundary dynamic masking/sampler state.
    Weighted replacement sampling changes draws, not stored Dataset or validation.
    """
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from .balance import balance_state,sampling_weights
    tc=train_config
    train_ds=DNADataset(train_records,tokenizer,label_to_id,model_config.max_length)
    val_ds=DNADataset(val_records,tokenizer,label_to_id,model_config.max_length)
    labels=[int(s['labels']) for s in train_ds.samples]
    weights,audit=balance_state(labels,len(label_to_id),tc)
    shuffle=torch.Generator().manual_seed(tc.seed)
    train_collator=DNACollator(tokenizer,tc.mlm_weight>0,tc.mlm_probability,tc.seed)
    val_collator=DNACollator(tokenizer,True,tc.mlm_probability,tc.seed+1000)
    sampler=None
    if tc.sampling=='weighted':
        sampler=WeightedRandomSampler(sampling_weights(labels,audit['training_class_counts'],tc.sampling_power),
            num_samples=tc.epoch_samples or len(train_ds),replacement=True,generator=shuffle)
    train=DataLoader(train_ds,batch_size=tc.batch_size,shuffle=sampler is None,
        sampler=sampler,collate_fn=train_collator,num_workers=0,generator=shuffle)
    def evaluation(ds,collator):
        return DataLoader(ds,batch_size=tc.batch_size,shuffle=False,collate_fn=collator,num_workers=0)
    return {'train':train,'validation':evaluation(val_ds,DNACollator(tokenizer)),
        'validation_mlm':evaluation(val_ds,val_collator),
        'train_clean':evaluation(train_ds,DNACollator(tokenizer)),
        'shuffle_generator':shuffle,'train_collator':train_collator,'val_collator':val_collator,
        'class_weights':weights,'balance':audit}
