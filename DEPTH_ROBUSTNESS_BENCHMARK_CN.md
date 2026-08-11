# NYUv2 深度退化鲁棒性基准

本基准只修改验证集的 Depth 输入，不修改 RGB、标签、模型或权重。退化发生在原始
8-bit 深度图读取之后、归一化与验证预处理之前。随机退化由“固定种子 + 样本名称”
确定，因此不同机器和重复运行能够得到相同退化结果。

## 协议

### pilot（5个条件）

用于快速检查作者权重与复现权重：Clean、30%随机缺失、Gaussian噪声0.03、水平错位
4像素、深度全零。

### core（18个条件）

- Clean；
- 随机像素缺失：10%、30%、50%；
- 单个连续方形区域缺失：64、128、192像素；
- Gaussian噪声：0.01、0.03、0.05（相对于8-bit深度范围）；
- Gaussian模糊：kernel 3、5、9；
- RGB-D水平错位：2、4、8像素；
- 完全缺失：全零填充、图像有效深度均值填充。

### representative（6个条件）

用于其他随机种子或其他模型规模的低成本趋势复核：Clean、30%/50%随机缺失、噪声
0.05、错位8像素和深度全零。它不是主表协议，不能替代core结果。

### extended（26个条件）

包含全部core条件，并增加：70%随机缺失、噪声0.10、错位16像素、深度尺度0.8/1.2、
随机异常深度1%/3%/5%。

主论文先使用core协议建立曲线；extended用于补充极端退化和错误深度分析。所有主表
默认采用单尺度、无翻转、AMP、validation batch size 1，以降低推理策略对鲁棒性结论
的干扰。

## 推荐运行顺序

先复用旧的5条件协议评估复现S权重，这样可以直接与已经完成的作者权重结果比较：

```bash
bash scripts/run_depth_robustness_nyuv2_s.sh \
  --gpu 1 \
  --checkpoint "checkpoints/NYUDepthv2_DFormerv2_S_20260722-170937/epoch-362_miou_56.43.pth" \
  --report-root validation_reports/depth_pilot_reproduced_s_seed12345
```

完成后与已有作者结果自动比较：

```bash
python scripts/compare_depth_robustness.py \
  --author-root validation_reports/depth_robustness_author_gpu1_20260723-150504 \
  --reproduced-root validation_reports/depth_pilot_reproduced_s_seed12345 \
  --output-dir validation_reports/depth_pilot_author_vs_reproduced
```

正式core协议同时运行作者与复现权重并自动比较：

```bash
bash scripts/run_nyuv2_s_depth_benchmark_pair.sh \
  --gpu 1 \
  --author-checkpoint "checkpoints/trained/NYU Depth v2/DFormerv2_Small_NYU.pth" \
  --reproduced-checkpoint "checkpoints/NYUDepthv2_DFormerv2_S_20260722-170937/epoch-362_miou_56.43.pth" \
  --protocol core \
  --report-root validation_reports/depth_benchmark_s_core_author_vs_reproduced
```

core结果确认无误后，可以在**同一个report-root**上把协议升级为extended。18个参数相同的
core条件会被复用，只新增运行8个扩展条件；如果退化代码本身发生变化，则增加`--rerun`
强制重跑。

脚本在前台管理所有子验证。若需要断开SSH后继续运行，应在外层使用nohup：

```bash
LOG=run_logs/depth_benchmark_s_core_gpu1.log
nohup bash scripts/run_nyuv2_s_depth_benchmark_pair.sh \
  --gpu 1 \
  --reproduced-checkpoint "checkpoints/NYUDepthv2_DFormerv2_S_20260722-170937/epoch-362_miou_56.43.pth" \
  --protocol core \
  --report-root validation_reports/depth_benchmark_s_core_author_vs_reproduced \
  > "$LOG" 2>&1 &
echo $!
```

查看进度：

```bash
tail -f run_logs/depth_benchmark_s_core_gpu1.log
```

相同命令可以安全重启。已存在且checkpoint匹配的完整JSON条件会被跳过；使用`--rerun`
才会强制重跑全部条件。

## 输出

每个权重目录包含：

- `depth_benchmark_manifest.json`：协议、权重、种子和条件定义；
- `depth_benchmark_summary.csv`：每个条件的总体指标和下降；
- `depth_benchmark_family_summary.csv`：每类退化的平均与最差结果；
- `depth_benchmark_per_class.csv`：40类别在所有条件下的长表；
- `depth_benchmark_aggregate.json`：整体鲁棒分数、最差条件和完整汇总；
- `<condition_id>/eval.log`：单条件完整日志；
- `<condition_id>/*.json`与`*.csv`：单条件总体、混淆矩阵和类别指标。

双权重脚本还会在`comparison/`生成条件、退化家族和类别级对比。

比较表中的`candidate_robustness_advantage`定义为：

```text
作者权重绝对下降 - 复现权重绝对下降
```

该值为正表示复现权重的性能下降更小，即相对更鲁棒。

## 其他复现权重的使用原则

第一轮正式基准只使用seed 12345的完整复现权重，因为该权重已经完成四种推理策略和
40类别分析，实验链条一致。seed 42与seed 3407不立即跑全部26个条件：待seed 3407
训练完成后，先在Clean、Missing-30%、Missing-50%、Noise-0.05、Shift-8和Zero等代表
条件上验证趋势。如果不同种子的下降一致，再把主要结论写入随机稳定性分析。B和L只有
在完成500 Epoch并通过Clean正式评估后，才用于模型规模鲁棒性对比。
