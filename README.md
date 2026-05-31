# Multi-Scale OT-based Process Drift Detection + LLM Root-Cause Evaluation

本项目实现一个端到端流程：

1. 对业务流程事件日志进行**多尺度漂移检测**（activity / DFG / trace 三个粒度的 Jensen-Shannon 散度）
2. 用**变体级 Optimal Transport**（Wasserstein-1 + Levenshtein 编辑距离 ground metric）做漂移归因，输出 case 级解释
3. 用 **PELT 换点检测** 定位漂移时刻，给出 95% bootstrap 置信区间
4. 用**置换检验**给出每个尺度的 p-value（替代手调阈值）
5. 用 LLM 自动生成中文 Markdown 诊断报告
6. 用注入式 ground truth 系统化**评估 LLM 根因解释质量**（precision/recall + oracle judge 三维 rubric）

---

## 1. 方法定位（vs 旧版）

旧版主线是 `max(TV(trace_dist), W(duration)/median) > 0.05`。新版直接面向"写论文"，从第一性原理重设计：

| Gap | 旧方法 | 新方法 |
|---|---|---|
| Aggregation gap | `max(TV, W)` 拼两个异质距离 | 三尺度 JSD 向量 + 变体级 W₁，分别带 p-value |
| Localization gap | 强制 50/50 切分 | PELT 换点检测 + bootstrap 95% CI |
| Attribution gap | 只能输出 top-K trace 频率 | OT 运输计划 π 给 case 级归因（"哪些 baseline case 流向了哪些 current case"）|
| Significance gap | 手调阈值 0.05 | 置换检验直接给 p-value（Phipson-Smyth 估计） |
| Interpretation gap | LLM 是末端工具 | LLM 是被评估的方法本身（带 ground truth 网格） |

---

## 2. 项目结构

```text
business-drift-detection/
├── drift/                      # 主线方法包（每个文件对应 Method 段一节）
│   ├── io.py                   # 事件日志加载 + per-case 表构造（原 convert_data.py）
│   ├── injection.py            # M5 Bose-style 漂移注入：insertion/deletion/substitution/loop
│   ├── metrics.py              # M1 三尺度 JSD（activity / DFG / trace）
│   ├── significance.py         # M4 置换检验 p-value
│   ├── localization.py         # M3 滑窗信号 + ruptures PELT + bootstrap CI
│   ├── ot_attribution.py       # M2 变体级 Wasserstein-1 + transport plan + case-level attribution
│   ├── evaluation.py           # M6 LLM 根因提取 + precision/recall + judge rubric
│   └── viz.py                  # M7 论文 figure 样式 + 7 张图的实现
│
├── run_full_pipeline.py        # 主流程编排（~330 行 thin orchestrator）
├── run_llm_evaluation.py       # LLM 评估网格 harness（pattern × seed，每格 3 个 LLM call）
├── make_figures.py             # 论文图 CLI：`--figure 1..7 | all`
├── convert_data.py             # 兼容 shim，重新导出 drift.io.*
│
├── tests/                      # pytest 103 cases
│   ├── test_smoke.py           # import 烟雾测试 + io 基础
│   ├── test_injection.py       # M5 单元测试
│   ├── test_metrics.py         # M1 单元测试
│   ├── test_significance.py    # M4 H0 均匀性 + H1 显著性
│   ├── test_localization.py    # M3 step/flat/double 信号
│   ├── test_ot_attribution.py  # M2 hand-computed transport + 端到端
│   └── test_evaluation.py      # M6 mocked-LLM 测试
│
├── datasets/
│   ├── finale.csv              # Help Desk 工单日志（4580 case / 14 活动 / 21k 事件）
│   ├── frequency-log.csv       # BPI Challenge 风格贷款流程（24 活动 / 71k 事件）
│   ├── frequency-log.xes       # 同上 XES 原文件
│   ├── llm_analyst_official.py # LLM 分析脚本（消费 schema v2，env-var 优先）
│   └── final_report_for_azure.json   # 主流程输出（schema v2）
│
├── examples/
│   └── Final_Drift_Analysis_Report.md   # LLM 输出报告示例
│
├── figures/                    # 7 张论文图 PDF（make_figures.py 生成）
├── outputs/                    # LLM 评估 grid JSON
├── archive/                    # 历史实验代码（非主线）
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 3. 环境与依赖

- Python 3.10+（已用 3.11 验证）
- 依赖见 `requirements.txt`：
  - 基础：`pandas, numpy, scipy, matplotlib`
  - LLM：`openai, httpx`
  - 新增：`pot>=0.9, ruptures>=1.1, editdistance>=0.8, seaborn>=0.13, pytest>=8.0`
- LLM 网关：API key/URL/model 从环境变量读取，缺省时落到脚本里的兼容值（请尽量用环境变量）：
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `OPENAI_MODEL`

安装：

```bash
pip install -r requirements.txt
```

---

## 4. 一键复现

```bash
# 0) 单元测试
pytest tests/

# 1) 主流程：默认 Help Desk + CPD 切分 + 置换检验
python run_full_pipeline.py
#   → 写出 datasets/final_report_for_azure.json（schema v2）
#         + final_llm_input_prompt.txt

# 2) LLM 分析（消费 schema v2 attribution + drift_vector）
python datasets/llm_analyst_official.py
#   → 写出 examples/Final_Drift_Analysis_Report.md

# 3) 7 张论文图
python make_figures.py --figure all
#   → figures/fig1..7_*.pdf

# 4) LLM 评估网格（4 pattern × 3 seed = 12 cell，每 cell 3 个 LLM call，约 10 min）
python run_llm_evaluation.py
#   → outputs/llm_evaluation_grid.json
```

---

## 5. CLI 与环境变量

### 5.1 `run_full_pipeline.py`

| 标志 | 说明 |
|---|---|
| `--legacy` | 跑 v1 `max(TV, W) > 0.05` 老路径（兼容旧 demo） |
| `--no-perm` | 跳过置换检验（更快但失去 p-value） |
| `--no-cpd` | 强制 50/50 切分（跳过 PELT） |

### 5.2 环境变量（运行时生效）

| 变量 | 默认 | 说明 |
|---|---|---|
| `EVENT_LOG_PATH` | `datasets/finale.csv` | 输入路径（支持 .csv/.xes/.xml） |
| `COL_CASE_ID` / `COL_ACTIVITY` / `COL_TIMESTAMP` | Help Desk 命名 | 列名映射 |
| `KEEP_ONLY_COMPLETE` | `true` | XES 只保留 `lifecycle:transition=complete` |
| `INJECT_DRIFT` | `false` | 是否注入漂移 |
| `DRIFT_PATTERN` | `insertion` | `insertion / deletion / substitution / loop` |
| `DRIFT_TARGET` | `Live Chat` | 注入目标活动 |
| `DRIFT_SECONDARY` | `AutoReview` | insertion 的 new_activity 或 substitution 的 dst |
| `DRIFT_FRACTION` | `0.5` | 受影响 case 比例 |
| `DRIFT_SEED` | `42` | 注入 + 置换 + bootstrap 随机种子 |
| `SPLIT_METHOD` | `cpd` | `cpd / midpoint` |
| `PERMUTATION_B` | `100` | 置换轮数 |

### 5.3 切到 BPI 2017

```bash
EVENT_LOG_PATH=datasets/frequency-log.csv \
  COL_CASE_ID="case:concept:name" \
  COL_ACTIVITY="concept:name" \
  COL_TIMESTAMP="time:timestamp" \
  python run_full_pipeline.py
```

### 5.4 启用注入并跑评估

```bash
INJECT_DRIFT=true DRIFT_PATTERN=insertion \
  DRIFT_TARGET="Take in charge ticket" DRIFT_SECONDARY=AutoReview \
  DRIFT_FRACTION=0.7 \
  python run_full_pipeline.py
```

---

## 6. Schema v2 — `datasets/final_report_for_azure.json`

```jsonc
{
  "schema_version": "2.0",
  "status": "DRIFT DETECTED" | "STABLE",
  "dataset": {
    "name": "finale", "n_events": 22909, "n_cases": 4580,
    "n_activities": 15, "n_variants_baseline": 158, "n_variants_current": 155,
    "n_variants_union": 269
  },
  "split": {
    "method": "cpd",
    "window": 200, "step": 50,
    "change_points": [2450],
    "ci_95": [[2200, 2550]]
  },
  "drift_vector": {
    "activity_jsd": 0.060, "activity_pvalue": 0.0099,
    "dfg_jsd":      0.231, "dfg_pvalue":      0.0099,
    "trace_jsd":    0.480,
    "trace_w1":     0.194,
    "aggregate_pvalue": 0.0099
  },
  "attribution": {
    "w1": 0.194,
    "top_transport_flows": [
      {
        "from_variant": ["Assign seriousness", "Take in charge ticket", "Resolve ticket", "Closed"],
        "to_variant":   ["Assign seriousness", "Take in charge ticket", "AutoReview", "Resolve ticket", "Closed"],
        "mass": 0.326, "edit_distance": 0.20,
        "from_case_ids_sample": ["Case 245", "Case 1300", ...],
        "to_case_ids_sample":   ["Case 2286", "Case 3900", ...]
      },
      ...
    ],
    "top_lost_variants":   [{"variant": [...], "mass_baseline": 0.50, "mass_current": 0.13, "delta": -0.37, ...}, ...],
    "top_gained_variants": [{"variant": [...], "mass_baseline": 0.00, "mass_current": 0.33, "delta": +0.33, ...}, ...]
  },
  "ground_truth": {
    "pattern": "insertion", "target_activity": "Take in charge ticket",
    "secondary_activity": "AutoReview", "fraction": 0.7, "seed": 42,
    "n_affected_cases": 1457, "affected_case_ids": [...],
    "description": "Insert 'AutoReview' after every 'Take in charge ticket' in 1457 cases (1561 new events)."
  } | null,
  "legacy": {
    "drift_score": 0.698, "trace_drift_score": 0.698,
    "duration_drift_score": 0.017, "drift_metric": "tv",
    "detection_threshold": 0.05, "status": "DRIFT DETECTED",
    "top_baseline_process_freq": {...}, "top_current_process_freq": {...},
    "baseline_count": 2450, "current_count": 2130
  }
}
```

旧版 LLM 脚本通过 `--legacy-prompt` 仍可消费 `legacy` block。

---

## 7. 7 张论文图（`make_figures.py --figure N`）

| # | 文件 | 内容 | 数据成本 |
|---|---|---|---|
| 1 | `fig1_teaser.pdf` | 多尺度漂移信号 + CPD/CI + top transport flows | ~10s（1 次注入 + CPD） |
| 2 | `fig2_method_overview.pdf` | 方法 pipeline 示意图（无数据） | <2s |
| 3 | `fig3_multi_scale.pdf` | 4 种注入模式下的三尺度 + W₁ 柱状（带 null 基线） | ~10s（4 次 pipeline） |
| 4 | `fig4_detection_roc.pdf` | proposed W₁ vs legacy max(TV,W) 的 ROC + AUC | ~40s（6 seed × 5 场景） |
| 5 | `fig5_localization.pdf` | 真实 vs 检测漂移点散点 + 延迟箱线图 | ~2-3 min（40 seed × 4 pattern） |
| 6 | `fig6_llm_rubric.pdf` | LLM judge 三维评分柱状（需先跑 `run_llm_evaluation.py`） | 占位/真实 |
| 7 | `fig7_ablation.pdf` | 组件消融：去掉 OT / trace / DFG / activity 后的 score | ~5s |

样式策略（`drift/viz.py:set_paper_style`）：seaborn `whitegrid` + `colorblind` palette + serif + 300dpi vector PDF。**proposed = 暖橙**，baseline = 冷灰，强调一致。

---

## 8. Tests

```bash
pytest tests/                       # 103 个测试，约 23s
pytest tests/test_significance.py   # H0 均匀性 + H1 显著性
pytest tests/test_ot_attribution.py # OT 两变体 hand-computed transport
```

测试涵盖：注入正确性 + seed 确定性、JSD 性质（identity / bounds / symmetry）、置换检验 H0 均匀性、CPD step/flat/double 信号、OT identity / hand-computed transport、LLM 评估的 mock-client 全流程。

---

## 9. Archive 目录

`archive/` 是历史实验代码集合（model verification、可视化原型、KS test demo 等），主线复现请优先使用 `run_full_pipeline.py` + `make_figures.py` + `run_llm_evaluation.py`。

---

## 10. 一句话复现

```bash
pytest tests/ && \
  python run_full_pipeline.py && \
  python datasets/llm_analyst_official.py && \
  python make_figures.py --figure all
```

执行完成查看：

- `datasets/final_report_for_azure.json`  ← schema v2 报告
- `examples/Final_Drift_Analysis_Report.md`  ← LLM 中文 Markdown 报告
- `figures/fig1..7_*.pdf`  ← 论文图

要完整跑出 fig6（LLM rubric）：

```bash
python run_llm_evaluation.py --seeds 3 --patterns insertion deletion substitution loop
python make_figures.py --figure 6
```
