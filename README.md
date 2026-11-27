# Python exercises final project

原题见 ASSIGNMENT.md。test1.ipynb 保留三题内容；第三题可独立运行。

## 文件入口

- dna_training/：数据、BPE、模型、训练和评价代码。
- configs/final.json：最终联合训练设置。
- final_model/model.pt：固定的第40轮最终模型；mlm_pretrained.pt 为复现实验所用MLM260初始化权重。
- prepared/：固定的划分、标签与BPE；reference_nt.fasta 为原始核苷酸参考文件。
- final_test/：全部1780条测试结果；final_test_igtr/：其中1718条IG/TR子集结果。
- reports/：最终PDF、Word、对比曲线与实验依据。
- runs/final/：最终联合训练的原始指标和TensorBoard事件。
- reference.fasta、cdr_sequences.tsv：第一题输入和结果。
- papers.csv、pubmed_shih_2010_2020.db、myc_papers.tsv：第二题数据和结果。

## 运行

使用 environment.yml 创建环境，并根据平台安装一个 requirements-torch 文件；详细步骤见 ENVIRONMENT.md。
在项目根目录打开 test1.ipynb。第三题默认加载固定模型并展示保存结果，不启动训练；第一二题保持原有代码，其中论文抓取需要网络。

复现联合训练：

```bash
python -m dna_training --config configs/final.json --action train --mode weights --checkpoint final_model/mlm_pretrained.pt
```

查看曲线：

```bash
tensorboard --logdir runs --port 6006
```

## 最终结果与范围

最终模型按最低验证分类损失选取，在测试前固定。完整测试准确率58.71%，IG/TR测试子集57.68%，Macro F1为0.3749。
混淆矩阵行是预测类别，列是真实类别。
历史训练含469条非IG/TR记录，详见 scope_audit.json；本项目保留已确定的模型和划分，不悄然替换为严格IG/TR训练。
历史中间实验已移出项目，不影响最终运行；清理记录见 CLEANUP.json。请勿根据已经公开的测试成绩反复选择模型。
