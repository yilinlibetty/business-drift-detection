# Business Drift Detection with Evidence-Driven LLM Diagnosis

本项目实现了一个从事件日志到业务诊断报告的完整流程，目标不是只回答“有没有 drift”，而是进一步回答：

- 漂移发生在时间线上的哪些区段
- 每个漂移点的主要变化是什么
- 哪些证据支持这些变化
- 候选根因是什么
- 有哪些后续改进建议和人工核验动作

当前系统的正式口径是：**输出的是证据支持的候选根因，不是未经验证的事实根因。**

---

## 1. 项目定位

这是一个面向业务流程事件日志的 drift detection 原型系统，适合：

- 毕业设计 / 论文原型
- 流程挖掘课程实验
- BPM 监控与诊断 demo
- “drift detection + LLM explanation” 方向的研究验证

与传统只输出一个 `DRIFT DETECTED / STABLE` 的方案不同，这个项目会继续生成：

- 漂移时间线
- 逐漂移点证据包
- 规则化候选根因标签
- 逐点诊断结果
- 总报告与人工核验模板

---

## 2. 当前方法概览

主流程默认采用 **timeline mode**。

### Step 1. 事件日志读取与标准化

系统支持：

- `.csv`
- `.xes`
- `.xml`

最少需要三列：

- `case id`
- `activity`
- `timestamp`

读取逻辑位于：

- `convert_data.py`

### Step 2. Case 聚合

系统会按 case 聚合出：

- `Trace`：完整活动路径
- `Duration`：case 总时长
- `EventCount`
- `RepeatedActivityCount`
- `HasLoop`

这一步把 event-level log 转成 case-level process behavior。

### Step 3. 时间线双窗口比较

系统不再使用固定前后 50/50 切分作为主模式，而是沿时间线构造相邻窗口：

- `reference window`
- `current window`

对每一对窗口计算：

- `trace_score`
  - trace 分布距离
  - 支持 `TV` 和 `L1`
- `duration_score`
  - case duration 的 Wasserstein 距离
  - 再除以 `median(reference)` 做归一化
- `final_score`
  - 默认取 `max(trace_score, duration_score)`

之后会对分数序列做 3 点中位数平滑。

### Step 4. 漂移区段识别

系统会基于阈值找出漂移窗口，然后合并成 `drift interval`。

当前策略包括：

- 超阈值窗口识别
- 相邻激活窗口成组
- 对彼此非常接近的组做二次合并，避免把同一个真实漂移切成多个点
- 对每个区段选取峰值窗口作为 `drift point`

输出字段包括：

- `interval_start_time`
- `interval_end_time`
- `peak_time`
- `peak_score`
- `trace_score`
- `duration_score`
- `detection_delay_proxy`

### Step 5. 证据提取

每个漂移点都会生成结构化 evidence pack，至少包括：

- `top_increased_traces`
- `top_decreased_traces`
- `top_changed_transitions`
- `activity_frequency_deltas`
- `rework_or_loop_rate_delta`
- `duration_stats_delta`
- `case_count`
- `window_time_range`
- `evidence_ids`

如果日志里存在更多业务字段，还会追加可选维度分析：

- `resource`
- `team`
- `priority`
- `channel`
- `region`

### Step 6. 规则化候选根因标签

在调用 LLM 之前，系统先根据证据打规则标签，例如：

- `path_added`
- `path_removed_or_skipped_step`
- `delay_increase`
- `loop_increase`
- `handoff_or_escalation_increase`
- `case_mix_shift`

这一步的目的是：

- 先给出确定性、可解释的候选方向
- 限制 LLM 自由发挥
- 让后续诊断更可控、更便于评估

### Step 7. LLM 或 fallback 诊断

如果配置了 LLM，系统会对每个漂移点单独做诊断；否则自动回退到规则驱动的 fallback 诊断。

诊断输出统一为结构化 JSON，至少包括：

- `summary`
- `candidate_causes[]`
- `recommendations[]`
- `confidence`
- `missing_data`

### Step 8. 报告渲染

最终系统会输出：

- 一份总报告 `final_drift_report.md`
- 一份人工核验模板 `human_review_rubric.md`
- 一份完整结构化分析结果 `drift_analysis.json`

---

## 3. 项目结构

```text
business-drift-detection/
├─ run_full_pipeline.py
├─ convert_data.py
├─ drift_detection/
│  ├─ pipeline.py
│  ├─ evidence.py
│  ├─ llm_support.py
│  └─ reporting.py
├─ datasets/
│  ├─ finale.csv
│  ├─ frequency-log.xes
│  ├─ frequency-log.csv
│  └─ llm_analyst_official.py
├─ outputs/
├─ archive/
├─ requirements.txt
└─ README.md
```

各模块职责：

- `run_full_pipeline.py`
  - CLI 入口
  - 编排整个检测、诊断、输出流程
- `drift_detection/pipeline.py`
  - 时间线检测
  - 阈值选择
  - 区段合并
  - 合成注入
  - evaluation
- `drift_detection/evidence.py`
  - 证据提取
  - 规则标签生成
- `drift_detection/llm_support.py`
  - LLM 调用
  - fallback 诊断
- `drift_detection/reporting.py`
  - Markdown 报告
  - 人工核验模板
- `datasets/llm_analyst_official.py`
  - 兼容 wrapper
  - 读取已有 `drift_analysis.json` 重新渲染报告

---

## 4. 环境准备

建议使用：

- `Python 3.9+`

安装依赖：

```bash
pip install -r requirements.txt
```

当前依赖：

- pandas
- numpy
- scipy
- matplotlib
- streamlit
- openai
- httpx
- pyyaml
- pytest

补充说明：

- 解析 XES/XML 时，`pm4py` 是可选依赖；没装也能跑。
- 未配置 LLM 时，主流程不会报错，而会自动使用 fallback 诊断。

---

## 5. 快速开始

### 5.1 默认运行

```powershell
python run_full_pipeline.py
```

默认行为：

- 使用 `timeline` 模式
- 自动阈值开启
- 若没有配置 `OPENAI_API_KEY`，自动使用 fallback 诊断

### 5.2 不调用 LLM，只生成证据驱动报告

```powershell
python run_full_pipeline.py --no-llm
```

### 5.3 运行多段合成漂移评估

```powershell
python run_full_pipeline.py --no-llm --inject-drift --drift-type mixed --drift-segments 2 --evaluate
```

补充说明：

- 若启用合成注入但没有显式传 `--detection-mode`，纯 `structure` 注入默认用 `structure` 检测，纯 `delay` 注入默认用 `delay` 检测。

### 5.4 使用兼容 wrapper 重新渲染报告

```powershell
python datasets\llm_analyst_official.py --no-llm
```

---

## 6. 重要 CLI 参数

### 分析相关

- `--analysis-mode timeline|legacy-half-split`
- `--legacy-half-split`
- `--window-size <int>`
- `--step-size <int>`
- `--threshold <float>`
- `--auto-threshold`
- `--fixed-threshold`
- `--top-k <int>`
- `--drift-metric tv|l1`
- `--detection-mode structure|delay|mixed|auto`
- `--output-dir <path>`

### LLM 相关

- `--llm-enabled`
- `--no-llm`

### 合成注入与评估相关

- `--inject-drift`
- `--no-inject-drift`
- `--drift-type structure|delay|mixed`
- `--drift-segments <int>`
- `--drift-segment-ratio <float>`
- `--target-activity <name>`
- `--drift-seed <int>`
- `--evaluate`

---

## 7. LLM 配置

当前不允许在代码里硬编码密钥。

请通过环境变量配置：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
$env:OPENAI_MODEL="gpt-5.2"
```

说明：

- 没有 `OPENAI_API_KEY` 时，系统自动走 fallback。
- `datasets/llm_analyst_official.py` 现在只是兼容入口，不再保存任何硬编码凭证。

---

## 8. 输出文件说明

所有正式输出都写入 `outputs/`。

### 8.1 `outputs/drift_analysis.json`

这是 canonical 输出，包含：

- `run_metadata`
- `config`
- `global_summary`
- `score_timeline`
- `drift_points[]`
- `ground_truth_intervals`
- `evaluation`
- `llm`

每个 `drift_point` 至少包含：

- `id`
- `interval_start_time`
- `interval_end_time`
- `peak_time`
- `peak_score`
- `trace_score`
- `duration_score`
- `evidence`
- `rule_based_tags`
- `llm_diagnosis`

### 8.2 `outputs/drift_score_timeline.csv`

窗口级分数输出，适合画时间线图和调阈值，包含：

- case 范围
- reference/current 时间范围
- `trace_score`
- `duration_score`
- `final_score_raw`
- `final_score`
- `threshold`
- `is_drift_window`

### 8.3 `outputs/final_drift_report.md`

总报告，固定结构为：

- 总览
- 检测方法与阈值
- 漂移时间线概览
- 逐漂移点分析
- 跨区段共性
- 改进建议
- Evaluation Snapshot（若存在）

### 8.4 `outputs/human_review_rubric.md`

人工核验模板，用于真实日志抽样评估：

- 变化是否真实
- 原因是否有证据支撑
- 建议是否可执行

### 8.5 `outputs/legacy_final_report_for_azure.json`

兼容旧字段的简化 JSON，方便旧流程迁移。

---

## 9. 评估指标

当存在合成 ground truth，或启用 `--evaluate` 时，系统会输出：

- `interval_level_precision`
- `interval_level_recall`
- `interval_level_f1`
- `false_positive_rate`
- `mean_detection_delay_cases`
- `cause_taxonomy_hit_rate`
- `evidence_fidelity`

这些指标的含义分别是：

- `interval_level_precision`
  - 预测出的漂移区段中，有多少是真的
- `interval_level_recall`
  - 真实漂移区段中，有多少被系统抓到
- `interval_level_f1`
  - precision / recall 的折中
- `false_positive_rate`
  - 稳定区段被误判为 drift 的比例
- `mean_detection_delay_cases`
  - 从真实漂移开始到系统首次命中该区段的 case 级延迟
- `cause_taxonomy_hit_rate`
  - 注入类型是否被候选标签正确命中
- `evidence_fidelity`
  - 报告引用的 `evidence_ids` 是否真实存在

---

## 10. 当前状态

目前系统已经能稳定完成：

- 多漂移点检测
- 漂移区段合并
- 逐点证据提取
- 候选根因标签生成
- fallback 诊断
- Markdown 总报告生成
- 区段级 evaluation

当前在合成结构漂移和 mixed 漂移场景下，区段级 precision / recall / F1 已经可以做到较好的结果。

---

## 11. 已知边界

在使用和写论文时，请注意以下边界：

- 候选根因是**证据支持的推断**，不是严格验证过的因果事实。
- 如果日志里只有 `case/activity/timestamp` 三列，系统在 `case_mix_shift`、责任转移验证等方面会比较保守。
- `legacy-half-split` 只是保留给对照实验，不是推荐主模式。
- 纯时长型 drift 的检测链路仍然比结构型 drift 更敏感于窗口和阈值设置，需要继续优化。

---

## 12. 推荐运行方式

```powershell
python run_full_pipeline.py --no-llm
python run_full_pipeline.py --no-llm --inject-drift --drift-type mixed --drift-segments 2 --evaluate
python datasets\llm_analyst_official.py --no-llm
```

运行后建议优先查看：

- `outputs/drift_analysis.json`
- `outputs/drift_score_timeline.csv`
- `outputs/final_drift_report.md`
- `outputs/human_review_rubric.md`

---

## 13. 论文冲刺版更新：multi-view scoring 与可复现实验

当前推荐的论文创新表述是：

> A configurable evidence-driven process drift diagnosis pipeline with multi-view scoring and reproducible ablation experiments.

也就是说，本项目不把候选根因表述为已经验证过的事实根因，而是输出**由 evidence ids 支撑的候选诊断假设**，并保留人工复核入口。

### 13.1 新增评分配置

主流程新增 `--score-profile`：

```powershell
python run_full_pipeline.py --no-llm --score-profile trace-duration
python run_full_pipeline.py --no-llm --score-profile multi-view
```

- `trace-duration`：兼容旧版行为，只使用 trace 分布漂移和 duration 漂移。
- `multi-view`：在 trace/duration 之外，额外计算 transition、loop/rework、attribute/case-mix 视角分数，并输出 `dominant_signal`。辅助视角会用 process signal 做锚定，并尊重 `detection_mode`，避免普通属性或结构波动在 delay-only 场景中单独触发漂移。

`outputs/drift_score_timeline.csv` 在 `multi-view` 下会包含：

- `transition_score`
- `loop_score`
- `attribute_score`
- `core_score`
- `dominant_signal`

每个 drift point 的 evidence pack 也会包含 `score_contribution`，用于解释触发漂移判断的主要视角。

### 13.2 自动阈值参数化

自动阈值公式为：

```text
threshold = max(configured_threshold, median(scores) + mad_multiplier * MAD(scores))
```

可通过 CLI 或环境变量配置：

```powershell
python run_full_pipeline.py --no-llm --mad-multiplier 2.5
$env:MAD_MULTIPLIER="2.5"
```

报告中的“检测方法与阈值”部分会显示 `mad_multiplier`。

### 13.3 规则配置外置

规则标签的阈值、confidence 和 escalation keywords 已外置到：

```text
config/tagging_rules.yaml
```

启动时会校验必要字段。如果缺少必须字段，系统会在导入 evidence 模块时给出清晰错误，而不是在运行中抛出不明确的 `KeyError`。

### 13.4 批量实验与 ablation

新增批量实验入口：

```powershell
python run_experiments.py
```

默认会比较：

- `trace-duration`
- `multi-view`

默认场景包括：

- `structure_1_segment`
- `delay_1_segment`
- `mixed_1_segment`
- `mixed_2_segments`

输出位置：

```text
outputs/experiments/experiment_summary.csv
outputs/experiments/experiment_summary.md
```

核心字段包括：

- `scenario`
- `seed`
- `score_profile`
- `precision`
- `recall`
- `f1`
- `false_positive_rate`
- `mean_detection_delay_cases`
- `taxonomy_hit_rate`
- `predicted_interval_count`

### 13.5 测试

新增 pytest 测试覆盖：

- 自动阈值和 `mad_multiplier`
- drift point 合并逻辑
- rule-based tagging
- YAML tagging rules 校验
- LLM fallback metadata
- Markdown report fallback 渲染
- multi-view score 非负性和 `dominant_signal`

运行方式：

```powershell
pytest -q
```

---

## 14. Streamlit frontend and analysis figures

新增研究原型前端入口：

```powershell
streamlit run streamlit_app.py
```

前端不做前后端分离，直接复用 `run_pipeline(config)`。当前 v1 支持：

- 上传 `.csv`、`.xes`、`.xml`，或直接使用默认 `datasets/finale.csv` 做 demo。
- 配置 case id、activity、timestamp、window size、step size、detection mode、score profile、drift metric、MAD multiplier、top-k 和 LLM 开关。
- 单次运行检测并在页面内保留结果。
- 展示 summary metrics、drift score timeline、drift point cards、简化 evidence details。
- 下载 `drift_analysis.json` 和 `final_drift_report.md`。
- 下载全部分析图 ZIP，或把图片保存到 `outputs/figures/`。

新增绘图模块位于：

```text
drift_detection/visualization.py
```

当前实现的核心图包括：

- Figure 1: drift score timeline，包含 final/trace/duration score、threshold 和 drift point 标注。
- Figure 2: trace distribution comparison，展示 reference window vs current window 的 Top-K trace 频率变化。
- Figure 3: activity frequency delta，展示 activity 增减方向。
- Figure 5: threshold sensitivity，展示不同 `mad_multiplier` 下检测到的 drift point 数量。

加分图按 evidence 可用性显示：

- Figure 4: duration comparison，优先使用 duration samples 画箱线图，否则退回 median/p90/mean summary bars。
- Figure 6: multi-view radar，基于 `score_contribution` 展示 trace/transition/duration/loop/attribute/core sub-score。
- Figure 7: score component heatmap，展示 timeline 上各 sub-score 的整体强弱。
- Figure 8: dominant signal distribution，展示 multi-view 下主要触发信号分布。
- Figure 9: transition delta，展示 transition 频率变化。
- Figure 10: attribute delta，展示 resource/team/priority/channel 等属性变化。
- Figure 11: drift point score breakdown，以柱状图展示单个漂移点的 sub-score 贡献。

如果已经生成了 `outputs/drift_analysis.json`，也可以不打开前端，直接命令行导出论文图：

```powershell
python export_figures.py
```

默认输出：

```text
outputs/figures/
```

支持导出多种格式：

```powershell
python export_figures.py --format png --format pdf
python export_figures.py --input outputs/drift_analysis.json --output-dir outputs/figures --format svg
```

前端展示的 LLM / fallback 诊断仍应理解为 evidence-supported candidate causes，不是已验证事实根因。

---

## 15. Presentation brief for PPT agent

本节用于直接交给 PPT 制作 agent。目标是帮助其快速理解项目叙事、选择图表、组织 slide，并避免把系统能力表述过度。

### 15.1 One-line pitch

本项目是一个面向业务流程事件日志的 drift detection 原型系统：它不仅检测流程是否发生漂移，还定位漂移区间、提取多视角证据，并生成可复现的候选诊断报告。

推荐英文表述：

> A configurable evidence-driven process drift diagnosis pipeline with multi-view scoring, reproducible experiments, and an interactive Streamlit analysis interface.

推荐中文表述：

> 一个可配置、证据驱动的业务流程漂移诊断系统，结合多视角评分、可复现实验和 Streamlit 可视化前端，支持从事件日志到候选根因报告的完整分析链路。

### 15.2 Core message

PPT 需要强调三点：

- 本项目不是只输出 `DRIFT DETECTED / STABLE`，而是输出漂移时间线、漂移点、证据包和候选诊断。
- `multi-view` scoring 是当前创新核心：除 trace/duration baseline 外，还引入 transition、loop/rework、attribute/case-mix 等视角。
- LLM 是可选解释层，不作为主实验依赖；默认 fallback 诊断可复现，且所有候选根因和建议都需要引用 evidence ids。

需要避免的表述：

- 不要说系统已经“证明了真实根因”。
- 不要说 LLM 诊断是 ground truth。
- 不要说所有 drift 类型都已经被完美解决；delay drift 仍对 window/threshold 更敏感。

推荐安全表述：

- `candidate root causes`
- `evidence-supported hypotheses`
- `diagnostic suggestions for human review`
- `not validated causal facts`

### 15.3 Suggested slide deck structure

建议做 12-15 页 PPT：

1. Title: Business Drift Detection with Evidence-Driven Diagnosis.
2. Motivation: 业务流程会随政策、系统、资源、客户组合发生变化，传统 drift detection 往往缺少可解释证据。
3. Problem Statement: 输入 event log，输出 drift intervals、evidence packs、candidate causes、recommendations、reports and figures。
4. System Overview: event log input -> case aggregation -> sliding windows -> scoring -> drift point detection -> evidence extraction -> rule/LLM diagnosis -> reports/figures/frontend。
5. Baseline Scoring: `trace-duration` profile，包含 `trace_score`、`duration_score` 和 `final_score`。
6. Multi-view Scoring: `trace_score`、`transition_score`、`duration_score`、`loop_score`、`attribute_score`、`core_score`、`dominant_signal`。
7. Thresholding and Drift Point Construction: MAD threshold 和 drift interval merging。
8. Evidence Pack: trace、transition、activity、duration、loop/rework、attribute delta，以及 evidence ids。
9. Diagnosis Layer: rule-based tags first，LLM optional，fallback reproducible，candidate causes must cite evidence ids。
10. Streamlit Prototype: upload/run/configure/view charts/download JSON/Markdown/figures ZIP。
11. Experimental Design: `structure`、`delay`、`mixed`、`mixed --drift-segments 2`，比较 `trace-duration` vs `multi-view`。
12. Key Figures: timeline、trace distribution、activity delta、threshold sensitivity、radar、heatmap、transition/attribute delta。
13. Results and Discussion: 使用 `outputs/figures/` 与 `outputs/experiments/` 中的结果，讨论 multi-view 优势和限制。
14. Reproducibility and Engineering: CLI、batch experiments、figure export、Streamlit、pytest。
15. Limitations and Future Work: 候选根因需业务验证，LLM 非主实验依赖，未来可做 online detection 和 human feedback loop。

### 15.4 Recommended figures for PPT

Figures can be exported by:

```powershell
python export_figures.py
```

If using the known working conda environment:

```powershell
D:\anaconda3\envs\nlp\python.exe export_figures.py
```

Default output folder:

```text
outputs/figures/
```

Recommended PPT figure mapping:

- Method overview slide: create a custom flow diagram from Section 15.3 slide 4.
- Drift detection result slide: `outputs/figures/figure_01_score_timeline.png`.
- Threshold robustness slide: `outputs/figures/figure_05_threshold_sensitivity.png`.
- Multi-view scoring slide: `outputs/figures/figure_07_score_component_heatmap.png`.
- Drift point evidence slide: `outputs/figures/dp01_figure_02_trace_distribution.png`.
- Activity evidence slide: `outputs/figures/dp01_figure_03_activity_delta.png`.
- Duration evidence slide: `outputs/figures/dp01_figure_04_duration_comparison.png`.
- Multi-view contribution slide: `outputs/figures/dp01_figure_06_multiview_radar.png`.
- Transition evidence slide: `outputs/figures/dp01_figure_09_transition_delta.png`.
- Attribute evidence slide: `outputs/figures/dp01_figure_10_attribute_delta.png`.
- Score breakdown slide: `outputs/figures/dp01_figure_11_score_breakdown.png`.

If a figure says no data is available, skip it for the PPT or explain that the corresponding evidence type was not present in the selected dataset/window.

### 15.5 Demo script for presentation

Recommended live demo flow:

1. Start the frontend:

```powershell
D:\anaconda3\envs\nlp\python.exe -m streamlit run streamlit_app.py
```

2. Do not upload a file.
3. Keep defaults: `datasets/finale.csv`, mode=`mixed`, score profile=`trace-duration`, auto threshold enabled, LLM disabled.
4. Click `Run drift detection`.
5. Show the summary cards and Figure 1 timeline.
6. Open a drift point card.
7. Open `Evidence details`.
8. Show trace, activity, duration and radar tabs.
9. Click `Download figures ZIP` or `Save figures to outputs/figures`.
10. End by opening the Markdown report or exported figure folder.

Recommended demo narration:

> The frontend is a research prototype, not a production monitoring system. It allows fast single-run analysis while reusing the same pipeline and evidence schema used by the CLI experiments.

### 15.6 Key artifacts and file paths

Core code:

- `run_full_pipeline.py`: CLI pipeline and `run_pipeline(config)`.
- `drift_detection/pipeline.py`: scoring, thresholding, drift point detection and evaluation.
- `drift_detection/evidence.py`: evidence pack and rule-based tags.
- `drift_detection/llm_support.py`: optional LLM diagnosis and fallback.
- `drift_detection/reporting.py`: Markdown report rendering.
- `drift_detection/visualization.py`: all Matplotlib figure functions and figure export helpers.
- `streamlit_app.py`: interactive frontend.
- `run_experiments.py`: batch ablation experiments.
- `export_figures.py`: export figures from `drift_analysis.json`.

Main outputs:

- `outputs/drift_analysis.json`: full structured result.
- `outputs/final_drift_report.md`: readable report.
- `outputs/drift_score_timeline.csv`: timeline scores.
- `outputs/experiments/experiment_summary.csv`: batch experiment table.
- `outputs/experiments/experiment_summary.md`: batch experiment summary.
- `outputs/figures/`: exported analysis figures.

### 15.7 Suggested visual style for slides

Suggested slide style:

- Use a clean research presentation style rather than a product marketing style.
- Prefer one key message per slide.
- Use timeline and evidence charts as the main visual anchors.
- Use callout boxes for `evidence ids`, `dominant_signal` and `candidate causes`.
- Keep the wording conservative: diagnosis is evidence-supported, not causally verified.

Suggested color semantics:

- Blue/teal: stable process signal or reference window.
- Orange/red: current window, drift interval or increased risk.
- Gray: fallback, caveat or missing data.

### 15.8 Short abstract for PPT intro

This project presents a configurable process drift diagnosis pipeline for business event logs. It transforms event-level logs into case-level behavior, compares adjacent timeline windows, detects drift intervals, and builds evidence packs for each drift point. Beyond a trace-duration baseline, the system supports multi-view scoring over trace, transition, duration, loop/rework and attribute/case-mix signals. A rule-based and optional LLM diagnosis layer converts evidence into candidate root-cause hypotheses and recommendations, while keeping outputs grounded through evidence ids. The project includes reproducible ablation experiments, a Streamlit frontend for interactive analysis, and exportable figures for reporting and presentation.
