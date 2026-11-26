import tempfile
import unittest
from pathlib import Path
import torch
from dna_training import ModelConfig, TrainConfig, train_experiment, load_model
from dna_training.data import DNADataset, DNACollator
from dna_training.model import DNATransformer

class Tokenizer:
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[MASK]']
    base_tokens = ['A','C','G','T','N']
    token_to_id = {t:i for i,t in enumerate(special_tokens + base_tokens)}
    merges = []
    def encode(self, sequence):
        return [self.token_to_id[t] for t in sequence]

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tok=Tokenizer()
        self.labels={'a':0,'b':1}
        self.train=[{'species':'a','sequence':'AAAA'}, {'species':'a','sequence':'AAAC'},
                    {'species':'b','sequence':'CCCC'}, {'species':'b','sequence':'CCCA'}]
        self.val=[{'species':'a','sequence':'AAAG'}, {'species':'b','sequence':'CCCG'}]
        self.mc=ModelConfig(d_model=8,nhead=2,num_layers=1,dim_feedforward=16,
                            dropout=0.0,max_length=16)
    def test_masking_preserves_answers_and_dataset(self):
        ds=DNADataset(self.train,self.tok,self.labels,max_length=16)
        original=ds[0]['input_ids'].clone()
        batch=DNACollator(self.tok,use_mlm=True,mlm_probability=1.0)([ds[0]])
        self.assertTrue(torch.equal(original,ds[0]['input_ids']))
        self.assertEqual(batch['mlm_labels'][0,0].item(),-100)
        self.assertTrue(torch.equal(batch['mlm_labels'][0,1:],original[1:]))
    def test_padding_does_not_change_cls(self):
        model=DNATransformer(9,2,0,self.mc).eval()
        a=torch.tensor([[2,4,5]])
        b=torch.tensor([[2,4,5,0,0]])
        with torch.no_grad():
            x=model(a,a.ne(0))['class_logits'];y=model(b,b.ne(0))['class_logits']
        self.assertTrue(torch.allclose(x,y,atol=1e-6))
    def test_roundtrip_resume_and_compatibility(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(max_epochs=1,batch_size=2,device='cpu',output_dir=d,
                            mlm_probability=0.0,train_clean_eval=True)
            first=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            loaded,checkpoint=load_model(first.last_checkpoint,device='cpu')
            self.assertEqual(checkpoint['epoch'],1)
            self.assertTrue((first.run_dir/'metrics.csv').exists())
            cfg.max_epochs=2
            second=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg,
                                    resume_from=first.last_checkpoint)
            self.assertEqual(len(second.history),2)
            self.assertEqual(second.history[-1]['epoch'],2)
            self.assertEqual(second.run_dir,first.run_dir)
            with self.assertRaises(ValueError):
                train_experiment(self.train,list(reversed(self.val)),self.tok,self.labels,
                                 self.mc,cfg,resume_from=second.last_checkpoint)
    def test_changed_training_goal_rejects_resume(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(max_epochs=1,batch_size=2,device='cpu',output_dir=d)
            first=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            cfg.max_epochs=2
            cfg.mlm_weight=0.0
            with self.assertRaisesRegex(ValueError, 'mlm_weight'):
                train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg,
                                 resume_from=first.last_checkpoint)

    def test_weights_mode_branches_without_reusing_optimizer(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(max_epochs=1,batch_size=2,device='cpu',output_dir=d)
            first=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            branch=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg,
                                   init_from=first.best_checkpoint)
            self.assertNotEqual(first.run_dir,branch.run_dir)
            self.assertEqual(len(branch.history),1)

    def test_epoch_boundary_resume_matches_uninterrupted(self):
        self.mc.dropout = 0.1
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(max_epochs=2,batch_size=2,device='cpu',output_dir=d,
                            train_clean_eval=False)
            full=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            cfg.max_epochs=1
            first=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            cfg.max_epochs=2
            resumed=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg,
                                     resume_from=first.last_checkpoint)
            a=torch.load(full.last_checkpoint,weights_only=True,map_location='cpu')
            b=torch.load(resumed.last_checkpoint,weights_only=True,map_location='cpu')
            for k,v in a['model_state_dict'].items():
                self.assertTrue(torch.equal(v,b['model_state_dict'][k]),k)


    def test_resume_repairs_interrupted_metrics_write(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=TrainConfig(max_epochs=1,batch_size=2,device='cpu',output_dir=d)
            first=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg)
            # A partially written / ahead-of-checkpoint CSV is only a derived report.
            with (first.run_dir/'metrics.csv').open('a') as f:
                f.write('2,partial interrupted row\n')
            cfg.max_epochs=2
            resumed=train_experiment(self.train,self.val,self.tok,self.labels,self.mc,cfg,
                                     resume_from=first.last_checkpoint)
            self.assertEqual([r['epoch'] for r in resumed.history],[1,2])
