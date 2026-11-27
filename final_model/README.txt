固定最终模型
文件 model.pt 为联合训练 E13 第40轮 best.pt 的逐字节副本。
选择依据是最低验证分类损失，测试前已固定，未按测试结果重选。
该模型使用 CLS 汇总，而非 mean pooling；分类头是 Linear(256,41)。
编码器为2层Transformer，d_model=256，8个注意力头，FFN=1024，正弦位置编码，词表128，最大输入512tokens（含CLS）。
训练来自纯MLM260轮模型，后续联合目标 L_cls + 0.3 L_mlm。推理为完整FASTA记录序列，禁用遮蔽和dropout。
权重内含模型配置、词表、标签映射等信息；配套 dna_training.load_model 可直接加载。
SHA256与选择协议见selection_manifest.json。测试完成后模型SHA256已再次核对一致。

评价范围
final_test 为原始保留测试集1780条，准确率58.71%。
final_test_igtr 为其中IG或TR前缀的1718条，准确率57.68%。
历史训练含469条非IG/TR记录，因此目标子集评价不能称为严格IG/TR专用训练。
基因前缀仅用于事后按题目范围分组，不作为模型输入或选择检查点的依据。
