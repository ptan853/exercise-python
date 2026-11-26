"""Server-friendly entrypoint: python -m dna_training --config configs/weighted.json."""
import argparse
from . import ExperimentConfig,prepare_data,load_prepared,run_experiment,evaluate_checkpoint,plot_training_curves

def main():
    parser=argparse.ArgumentParser(description='Prepare, train or evaluate a DNA experiment')
    parser.add_argument('--config',required=True)
    parser.add_argument('--action',choices=['prepare','train','evaluate','plot'],default='train')
    parser.add_argument('--prepared',help='Existing prepared dataset directory; no refitting')
    parser.add_argument('--mode',choices=['fresh','resume','weights'],default='fresh')
    parser.add_argument('--checkpoint')
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--run-dir',help='Run directory for plotting')
    args=parser.parse_args();config=ExperimentConfig.load(args.config)
    if args.action=='plot':
        if not args.run_dir:parser.error('--run-dir required for plot')
        print(plot_training_curves(args.run_dir));return
    if args.action=='train':
        if args.mode!='fresh' and not args.checkpoint:parser.error('--checkpoint required for resume/weights')
        if args.mode=='fresh' and args.checkpoint:parser.error('fresh must not include --checkpoint')
    if args.action=='evaluate' and not args.checkpoint:parser.error('--checkpoint required for evaluate')
    prepared=load_prepared(args.prepared) if args.prepared else prepare_data(config.data,config.bpe)
    print('Prepared:',prepared.path,flush=True)
    if args.action=='prepare':return
    if args.action=='evaluate':
        evaluation=evaluate_checkpoint(prepared,args.checkpoint,split=args.split,device=config.training.device,
                                       batch_size=config.training.batch_size)
        print(evaluation.metrics);print(evaluation.output_dir);return
    result=run_experiment(prepared,config,
        resume_from=args.checkpoint if args.mode=='resume' else None,
        init_from=args.checkpoint if args.mode=='weights' else None)
    print('Best:',result.best_checkpoint);print(plot_training_curves(result.run_dir))

if __name__=='__main__':main()
