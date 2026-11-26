import csv,json,tempfile,unittest
from pathlib import Path
import torch
from dna_training import (DataConfig,BPEConfig,ExperimentConfig,prepare_data,
    load_prepared,DNABytePairEncoder,ModelConfig,TrainConfig,run_experiment,
    evaluate_checkpoint,plot_training_curves,compare_experiments)
from dna_training.balance import class_weights,sampling_weights

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        f=self.root/'input.fasta';lines=[]
        for label,seqs in [('Species a',['AAAA','AAAC','AAAG','AAAT','AACA','AACC']),
                           ('Species b',['CCCC','CCCA','CCCG','CCCT','CCGC','CCGG'])]:
            for i,seq in enumerate(seqs):
                lines += [f'>ID{i}|TRAV{i}*01|{label}_individual|F|V-REGION|',seq]
        f.write_text('\n'.join(lines)+'\n');self.f=f
        self.dc=DataConfig(source_path=str(f),cache_dir=str(self.root/'cache'))
        self.bc=BPEConfig(vocab_size=16,min_frequency=2)
    def tearDown(self):self.tmp.cleanup()
    def test_bpe_roundtrip_and_unknown(self):
        t=DNABytePairEncoder().fit(['ATGCATGC','ATGCATGA'],vocab_size=16)
        p=self.root/'tokenizer.json';t.save(p);loaded=DNABytePairEncoder.load(p)
        self.assertEqual(t.encode('ATGR'),loaded.encode('ATGR'))
        self.assertEqual(''.join(loaded.tokenize('ATGR')),'ATGN')
    def test_cached_split_and_separate_tokenizer(self):
        a=prepare_data(self.dc,self.bc);b=load_prepared(a.path)
        self.assertEqual(a.train_records,b.train_records)
        self.assertEqual(a.tokenizer.token_to_id,b.tokenizer.token_to_id)
        c=prepare_data(self.dc,BPEConfig(vocab_size=20,min_frequency=2))
        self.assertEqual(a.split_id,c.split_id)
        self.assertNotEqual(a.prepared_id,c.prepared_id)
        self.assertEqual(a.train_records,c.train_records)
    def test_balance_math(self):
        counts=torch.tensor([9,1])
        w=class_weights(counts,'inverse_sqrt',cap=100)
        self.assertAlmostEqual((w[1]/w[0]).item(),3,places=5)
        s=sampling_weights([0]*9+[1],counts,power=1.)
        self.assertAlmostEqual(s[:9].sum().item(),s[9].item(),places=5)
    def test_cache_tampering_is_detected(self):
        p=prepare_data(self.dc,self.bc)
        state=json.loads((p.path/'tokenizer.json').read_text())
        state['token_to_id']['A'],state['token_to_id']['C']=state['token_to_id']['C'],state['token_to_id']['A']
        (p.path/'tokenizer.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'modified'):
            load_prepared(p.path)

    def test_weighted_loss_math_and_clean_validation(self):
        from dna_training.trainer import _losses
        logits=torch.tensor([[2.,0.],[0.,1.],[0.,2.]],requires_grad=True)
        labels=torch.tensor([0,0,1])
        output={'class_logits':logits,'mlm_logits':torch.zeros(3,2,9,requires_grad=True)}
        batch={'labels':labels,'mlm_labels':torch.full((3,2),-100)}
        weights=torch.tensor([1.,3.])
        total,raw,mlm,n,objective=_losses(output,batch,0.,weights)
        ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
        self.assertTrue(torch.allclose(raw,ce.mean()))
        self.assertTrue(torch.allclose(total,(ce*weights[labels]).sum()/weights[labels].sum()))
        clean=_losses(output,batch,0.)[0]
        self.assertTrue(torch.allclose(clean,ce.mean()))
        total.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_sampler_resume_with_dropout(self):
        from dna_training import train_experiment
        p=prepare_data(self.dc,self.bc)
        mc=ModelConfig(d_model=8,nhead=2,num_layers=1,dim_feedforward=16,max_length=32,dropout=.1)
        tc=TrainConfig(max_epochs=2,batch_size=3,device='cpu',output_dir=str(self.root/'runs'),
            class_weighting='inverse_sqrt',sampling='weighted',train_clean_eval=False)
        full=train_experiment(p.train_records,p.val_records,p.tokenizer,p.label_to_id,mc,tc)
        tc.max_epochs=1
        first=train_experiment(p.train_records,p.val_records,p.tokenizer,p.label_to_id,mc,tc)
        tc.max_epochs=2
        resumed=train_experiment(p.train_records,p.val_records,p.tokenizer,p.label_to_id,mc,tc,
            resume_from=first.last_checkpoint)
        a=torch.load(full.last_checkpoint,map_location='cpu',weights_only=True)
        b=torch.load(resumed.last_checkpoint,map_location='cpu',weights_only=True)
        for key,value in a['model_state_dict'].items():
            self.assertTrue(torch.equal(value,b['model_state_dict'][key]),key)

    def test_end_to_end_eval_orientation_and_plots(self):
        p=prepare_data(self.dc,self.bc)
        cfg=ExperimentConfig(data=self.dc,bpe=self.bc,
            model=ModelConfig(d_model=8,nhead=2,num_layers=1,dim_feedforward=16,max_length=32,dropout=0),
            training=TrainConfig(max_epochs=1,batch_size=4,device='cpu',mlm_weight=0,
                class_weighting='inverse_sqrt',sampling='weighted',output_dir=str(self.root/'runs')))
        config_path=self.root/'experiment.json';cfg.save(config_path)
        self.assertEqual(cfg,ExperimentConfig.load(config_path))
        r=run_experiment(p,cfg)
        e=evaluate_checkpoint(p,r.best_checkpoint,device='cpu')
        self.assertEqual(e.metrics['samples'],len(p.val_records))
        self.assertEqual(int(e.confusion_pred_true.sum()),len(p.val_records))
        self.assertEqual(e.confusion_pred_true.sum(0).tolist(),
            torch.bincount(torch.tensor(e.y_true),minlength=2).tolist())
        self.assertTrue((e.output_dir/'confusion_matrix.png').exists())
        self.assertTrue(plot_training_curves(r.run_dir).exists())
        self.assertTrue(compare_experiments([r.run_dir],self.root/'comparison.png').exists())
        with self.assertRaisesRegex(ValueError,'split'):
            evaluate_checkpoint(p,r.best_checkpoint,split='typo',device='cpu')
        balance=json.loads((r.run_dir/'balance.json').read_text())
        self.assertEqual(balance['sampling'],'weighted')
        self.assertTrue(balance['class_weights'])

if __name__=='__main__':unittest.main()
