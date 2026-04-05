# Automated Business Process Drift Detection & LLM Root Cause Analysis

本项目实现了一个端到端流程：
1. 对业务流程事件日志进行漂移检测（结构漂移 + 时长漂移）
2. 输出结构化 JSON 证据
3. 交给大模型自动生成中文诊断报告（Markdown）

适用于毕业设计、流程挖掘课程实验、以及流程监控原型验证。

---

## 1. 项目目标

传统漂移检测通常只能回答“有没有漂移”。本项目进一步回答：
- 漂移有多强（量化分数）
- 主要变化发生在哪些流程路径上
- 可能根因是什么
- 后续可执行改进建议是什么

---

## 2. 项目结构

```text
FYP/
├─ run_full_pipeline.py              # 主流程：读取数据、构建Trace、检测漂移、输出JSON/Prompt
├─ convert_data.py                   # 日志转换器：支持 CSV / XES 读取，XES 可转 CSV
├─ final_llm_input_prompt.txt        # 主流程生成的给 LLM 的提示词（运行后更新）
├─ requirements.txt                  # 运行依赖
├─ README.md                         # 项目说明（本文件）
├─ datasets/
│  ├─ finale.csv                     # 默认事件日志数据（Help Desk）
│  ├─ frequency-log.xes              # XES 示例数据
│  ├─ frequency-log.csv              # 由 XES 转换后的 CSV
│  ├─ final_report_for_azure.json    # 主流程输出：漂移检测结果
│  └─ llm_analyst_official.py        # LLM 分析脚本：读取 JSON，生成 Markdown 报告
├─ examples/
│  └─ Final_Drift_Analysis_Report.md # LLM 输出报告示例
└─ archive/                          # 历史实验代码（非主线）
```

---

## 3. 核心流程（主线）

### Step A: 漂移检测（`run_full_pipeline.py`）

主脚本完成以下工作：
- 读取事件日志（默认 `datasets/finale.csv`，也支持 `.xes/.xml`）
- 清洗并按 `Case ID` 聚合为 Trace
- 按案例时间一分为二：前 50% 为 Baseline，后 50% 为 Current
- 可选注入人工漂移（结构缺失 / 延迟 / 混合）
- 计算漂移分数：
  - 结构漂移：Trace 分布距离（TV 或 L1）
  - 时长漂移：Wasserstein 距离并按基准中位数归一化
  - 最终分数：根据模式取结构、时长或两者 max
- 输出：
  - `datasets/final_report_for_azure.json`
  - `final_llm_input_prompt.txt`

### Step B: 报告生成（`datasets/llm_analyst_official.py`）

该脚本读取 `datasets/final_report_for_azure.json`，调用 OpenAI 兼容接口，生成中文 Markdown 报告：
- 输出文件：`examples/Final_Drift_Analysis_Report.md`
- 固定报告结构：`总览 / 关键变化 / 根因推断 / 改进建议`

---

## 4. 环境准备

### 4.1 Python 版本

建议 `Python 3.10+`（本仓库环境验证过 `Python 3.12`）。

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：
- pandas
- numpy
- scipy
- matplotlib
- openai
- httpx

说明：
- 读取 XES 时 `pm4py` 是可选依赖；未安装时会走内置 XML 解析逻辑。

---

## 5. 快速开始

### 5.1 运行检测主流程

Windows + 项目自带虚拟环境示例：

```powershell
.venv\Scripts\python.exe run_full_pipeline.py
```

成功后会看到：
- 漂移分数输出（Trace / Duration / Final）
- `datasets/final_report_for_azure.json` 写入成功
- `final_llm_input_prompt.txt` 生成成功

### 5.2 运行评估模式（可选）

```powershell
.venv\Scripts\python.exe run_full_pipeline.py --evaluate
```

可选参数：

```text
--eval-window <int>      # 默认 500
--eval-step <int>        # 默认 250
--eval-threshold <float> # 默认 DRIFT_THRESHOLD（0.05）
--auto-threshold         # 根据窗口评分自动选 best_threshold
--eval-seed <int>        # 默认 42
```

评估模式额外输出：
- `datasets/evaluation_report.json`
- 并把 `evaluation` 字段回写到 `datasets/final_report_for_azure.json`

### 5.3 生成 LLM 分析报告

```powershell
.venv\Scripts\python.exe datasets\llm_analyst_official.py
```

输出：
- `examples/Final_Drift_Analysis_Report.md`

---

## 6. 配置说明

### 6.1 主脚本环境变量（`run_full_pipeline.py`）

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `EVENT_LOG_PATH` | `datasets/finale.csv` | 输入日志路径（支持 `.csv/.xes/.xml`） |
| `COL_CASE_ID` | `Case ID` | 案例 ID 列名 |
| `COL_ACTIVITY` | `Activity` | 活动列名 |
| `COL_TIMESTAMP` | `Complete Timestamp` | 时间戳列名 |
| `KEEP_ONLY_COMPLETE` | `true` | XES 解析时是否只保留 `lifecycle:transition=complete` |
| `INJECT_DRIFT` | `false` | 是否在 Current 部分注入人工漂移 |
| `DRIFT_SEED` | `42` | 注入/评估随机种子 |
| `TOP_K_TRACES` | `10` | 报告中 Top-K 路径数量 |
| `DRIFT_METRIC` | `tv` | 结构分布距离：`tv` 或 `l1` |
| `DRIFT_THRESHOLD` | `0.05` | 漂移判定阈值 |
| `DETECTION_MODE` | `structure` | `structure` / `delay` / `mixed` / `auto` |

脚本中的固定参数（非环境变量）：
- `DRIFT_TYPE = 'structure'`（注入类型）
- `TARGET_ACTIVITY = 'Live Chat'`（延迟注入目标活动）

### 6.2 常用运行示例

#### 示例 A：切到 XES 数据并映射列名

```powershell
$env:EVENT_LOG_PATH="datasets/frequency-log.xes"
$env:COL_CASE_ID="case:concept:name"
$env:COL_ACTIVITY="concept:name"
$env:COL_TIMESTAMP="time:timestamp"
.venv\Scripts\python.exe run_full_pipeline.py
```

#### 示例 B：启用人工漂移注入

```powershell
$env:INJECT_DRIFT="true"
$env:DRIFT_SEED="42"
.venv\Scripts\python.exe run_full_pipeline.py
```

#### 示例 C：切换检测模式和度量

```powershell
$env:DETECTION_MODE="mixed"
$env:DRIFT_METRIC="l1"
$env:DRIFT_THRESHOLD="0.08"
.venv\Scripts\python.exe run_full_pipeline.py
```

---

## 7. 输出文件说明

### 7.1 `datasets/final_report_for_azure.json`

核心字段：
- `status`: `DRIFT DETECTED` / `STABLE`
- `drift_score`: 最终判定分数
- `trace_drift_score`: 结构漂移分数
- `duration_drift_score`: 归一化时长漂移分数
- `duration_drift_score_raw`: 原始 Wasserstein 距离
- `drift_metric`: `tv` / `l1`
- `detection_mode`: 判定模式
- `detection_threshold`: 阈值
- `analysis`:
  - `baseline_count`, `current_count`
  - `top_baseline_process_freq`, `top_current_process_freq`
  - `top_baseline_process_count`, `top_current_process_count`

若启用 `--evaluate`，还会增加：
- `evaluation`（包含混淆矩阵、precision/recall/f1、best_threshold 等）

### 7.2 `final_llm_input_prompt.txt`

给大模型直接使用的完整 Prompt，包含：
- 系统状态
- 漂移分数
- Baseline/Current Top-K 路径统计
- 固定报告输出要求

### 7.3 `examples/Final_Drift_Analysis_Report.md`

最终业务可读报告（中文 Markdown）。

---

## 8. LLM 脚本配置与安全提示

`datasets/llm_analyst_official.py` 中当前使用了脚本内配置项：
- `API_KEY`
- `MODEL_NAME`
- `BASE_URL`

建议在本地私有环境中使用，避免把真实密钥提交到仓库。推荐做法：
1. 把密钥改为环境变量读取
2. 把密钥文件加入 `.gitignore`
3. 若密钥已暴露，立即在服务端轮换

---

## 9. 常见问题（Troubleshooting）

### 9.1 `ModuleNotFoundError: No module named 'pandas'`

原因：使用了系统 Python 而不是项目虚拟环境。  
解决：

```powershell
.venv\Scripts\python.exe run_full_pipeline.py
```

### 9.2 `Missing columns` 报错

原因：输入数据列名和默认映射不一致。  
解决：设置 `COL_CASE_ID/COL_ACTIVITY/COL_TIMESTAMP` 环境变量为真实列名。

### 9.3 LLM 脚本调用失败（网络/认证）

可能原因：
- API Key 无效
- Base URL 不可访问
- 网络限制

解决：
- 校验 `API_KEY` / `BASE_URL`
- 先确认 `datasets/final_report_for_azure.json` 已生成
- 再单独运行 `datasets/llm_analyst_official.py`

---

## 10. Archive 目录说明

`archive/` 为历史实验代码集合（模型验证、可视化、早期原型），例如：
- `verify_model.py`: 合成数据验证检测灵敏度
- `run_drift_test.py`: 基于 KS 检验的示例
- `run_real_analysis.py`: 旧版真实数据分析脚本

主线复现建议优先使用：
- `run_full_pipeline.py`
- `datasets/llm_analyst_official.py`

---

## 11. 一句话复现

```powershell
.venv\Scripts\python.exe run_full_pipeline.py
.venv\Scripts\python.exe datasets\llm_analyst_official.py
```

执行完成后查看：
- `datasets/final_report_for_azure.json`
- `examples/Final_Drift_Analysis_Report.md`
