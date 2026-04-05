import pandas as pd
import numpy as np
import json
import uuid
import os
import random
import argparse
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from convert_data import load_event_log

# ================= 配置区域 =================
# 确保这里是您的真实文件路径 (建议使用相对路径)
FILE_PATH = os.getenv(
    "EVENT_LOG_PATH",
    os.path.join(os.path.dirname(__file__), "datasets", "finale.csv"),
)

# 字段映射 (默认 finale.csv，可通过环境变量覆盖)
COL_CASE_ID = os.getenv("COL_CASE_ID", "Case ID")
COL_ACTIVITY = os.getenv("COL_ACTIVITY", "Activity")
COL_TIMESTAMP = os.getenv("COL_TIMESTAMP", "Complete Timestamp")
KEEP_ONLY_COMPLETE = os.getenv("KEEP_ONLY_COMPLETE", "true").lower() in {"1", "true", "yes"}

# 漂移注入配置
INJECT_DRIFT = os.getenv("INJECT_DRIFT", "false").lower() in {"1", "true", "yes"}  # 开关
DRIFT_TYPE = 'structure'  # 可选: 'delay' (时间), 'structure' (流程), 'mixed' (混合)
TARGET_ACTIVITY = 'Live Chat'  # 针对哪个活动注入延迟 (示例)
DRIFT_SEED = int(os.getenv("DRIFT_SEED", "42"))

# 检测窗口配置
WINDOW_SIZE = 300  # 窗口大小
THRESHOLD_SENSITIVITY = 2.0  # 阈值倍数 (Mean + n * Std)
TOP_K_TRACES = int(os.getenv("TOP_K_TRACES", "10"))
DRIFT_METRIC = os.getenv("DRIFT_METRIC", "tv").lower()  # tv | l1
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
DETECTION_MODE = os.getenv("DETECTION_MODE", DRIFT_TYPE).lower()  # structure | delay | mixed | auto


# ===========================================

def compute_distribution_distance(vec_a, vec_b, metric=DRIFT_METRIC):
    """Compute distance between two discrete distributions."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if metric == "l1":
        return float(np.sum(np.abs(a - b)))
    # Total variation distance
    return float(0.5 * np.sum(np.abs(a - b)))


def compute_duration_drift(baseline_cases, current_cases):
    baseline = baseline_cases['Duration'].dropna().to_numpy(dtype=float)
    current = current_cases['Duration'].dropna().to_numpy(dtype=float)
    if baseline.size == 0 or current.size == 0:
        return 0.0, 0.0
    raw = float(wasserstein_distance(baseline, current))
    scale = float(np.median(baseline)) if np.median(baseline) > 0 else float(np.mean(baseline))
    if scale <= 0:
        return raw, raw
    return raw / scale, raw


def combine_drift_scores(trace_score, duration_score, mode=DETECTION_MODE):
    if mode == "structure":
        return trace_score
    if mode == "delay":
        return duration_score
    # mixed or auto -> be conservative and take the max
    return max(trace_score, duration_score)


class PipelineProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cases = None

    def step1_load_and_preprocess(self):
        """1. 加载数据并清洗"""
        print(f"\n[Step 1] Loading data from {self.file_path}...")
        try:
            self.df = load_event_log(
                self.file_path,
                COL_CASE_ID,
                COL_ACTIVITY,
                COL_TIMESTAMP,
                keep_only_complete=KEEP_ONLY_COMPLETE,
            )
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.file_path}")
            return False
        except Exception as exc:
            print(f"❌ Error: Failed to load log: {exc}")
            return False

        # 转换时间格式
        missing_cols = [c for c in [COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP] if c not in self.df.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns: {missing_cols}")
            print(f"      -> Available columns: {list(self.df.columns)}")
            return False

        self.df[COL_TIMESTAMP] = pd.to_datetime(self.df[COL_TIMESTAMP])
        self.df = self.df.sort_values(COL_TIMESTAMP)

        print(f"      -> Loaded {len(self.df)} events.")
        return True

    def step2_generate_traces(self, df_subset):
        """2. 将 Event Log 组合成 Trace"""
        # 按照 Case ID 聚合，形成 Trace 字符串
        # 同时计算该 Case 的总耗时 (End - Start)
        cases = df_subset.groupby(COL_CASE_ID).agg({
            COL_ACTIVITY: lambda x: ' -> '.join(map(str, x)),
            COL_TIMESTAMP: ['min', 'max', 'count']  # 记录开始时间、结束时间、事件数
        }).reset_index()

        # 展平列名
        cases.columns = ['CaseID', 'Trace', 'StartTime', 'EndTime', 'EventCount']
        cases['Duration'] = (cases['EndTime'] - cases['StartTime']).dt.total_seconds() / 60.0  # 分钟

        # 按结束时间排序
        cases = cases.sort_values('EndTime')
        return cases

    def step3_inject_drift(self, df_target):
        """3. 人为注入漂移 (模拟异常)"""
        if not INJECT_DRIFT:
            return df_target

        print(f"\n[Step 3] Injecting Artificial Drift ({DRIFT_TYPE})...")
        df_mod = df_target.copy()

        # --- 场景 A: 时间延长 (模拟效率降低) ---
        if DRIFT_TYPE in ['delay', 'mixed']:
            # 逻辑: 找到所有 'Target Activity'，将其时间戳往后推，模拟处理变慢
            mask = df_mod[COL_ACTIVITY] == TARGET_ACTIVITY
            target_cases = df_mod.loc[mask, COL_CASE_ID].unique().tolist()
            if target_cases:
                # 随机选择 80% 的 Case 进行延迟
                sample_size = max(1, int(len(target_cases) * 0.8))
                delayed_cases = set(np.random.choice(target_cases, size=sample_size, replace=False))
                delayed_events = 0
                for case_id in delayed_cases:
                    case_mask = df_mod[COL_CASE_ID] == case_id
                    target_times = df_mod.loc[case_mask & mask, COL_TIMESTAMP]
                    if target_times.empty:
                        continue
                    shift_start = target_times.min()
                    delay_minutes = int(np.random.randint(30, 120))
                    shift_mask = case_mask & (df_mod[COL_TIMESTAMP] >= shift_start)
                    df_mod.loc[shift_mask, COL_TIMESTAMP] = (
                        df_mod.loc[shift_mask, COL_TIMESTAMP] + pd.to_timedelta(delay_minutes, unit="m")
                    )
                    delayed_events += int(shift_mask.sum())
                print(f"      -> Injected time delays into {len(delayed_cases)} cases, {delayed_events} events shifted.")
            else:
                print("      -> No target activity found for delay injection.")

        # --- 场景 B: 结构缺失 (模拟跳过步骤) ---
        if DRIFT_TYPE in ['structure', 'mixed']:
            # 逻辑: 随机删除某些行，模拟步骤缺失
            original_len = len(df_mod)
            # 假设只有 10% 的概率会发生步骤丢失
            df_mod = df_mod.drop(df_mod.sample(frac=0.1).index)
            print(f"      -> Dropped {original_len - len(df_mod)} events to simulate missing steps.")

        # 重新排序，确保每个 Case 内部事件时间顺序正确
        df_mod = df_mod.sort_values([COL_CASE_ID, COL_TIMESTAMP])
        return df_mod

    def step4_detect_and_report(self, baseline_cases, current_cases):
        """4. 检测漂移并生成报告"""
        print(f"\n[Step 4] Running Detection Pipeline...")

        # 准备向量化
        all_traces = sorted(set(baseline_cases['Trace'].unique()) | set(current_cases['Trace'].unique()))
        trace_map = {t: i for i, t in enumerate(all_traces)}

        def get_vec(cases_df):
            counts = cases_df['Trace'].value_counts(normalize=True)
            vec = np.zeros(len(all_traces))
            for t, freq in counts.items():
                vec[trace_map[t]] = freq
            return vec

        # 计算分布
        baseline_vec = get_vec(baseline_cases)
        current_vec = get_vec(current_cases)

        # 计算距离
        trace_score = compute_distribution_distance(baseline_vec, current_vec)
        duration_score, duration_score_raw = compute_duration_drift(baseline_cases, current_cases)
        score = combine_drift_scores(trace_score, duration_score)
        print(f"      -> Trace Drift Score ({DRIFT_METRIC.upper()}): {trace_score:.4f}")
        print(f"      -> Duration Drift Score (Wasserstein/Median): {duration_score:.4f}")
        print(f"      -> Final Drift Score ({DETECTION_MODE.upper()}): {score:.4f}")

        # 阈值判定
        is_drift = score > DRIFT_THRESHOLD

        top_baseline_freq = baseline_cases['Trace'].value_counts(normalize=True).head(TOP_K_TRACES).to_dict()
        top_current_freq = current_cases['Trace'].value_counts(normalize=True).head(TOP_K_TRACES).to_dict()
        top_baseline_count = baseline_cases['Trace'].value_counts().head(TOP_K_TRACES).to_dict()
        top_current_count = current_cases['Trace'].value_counts().head(TOP_K_TRACES).to_dict()

        report = {
            "status": "DRIFT DETECTED" if is_drift else "STABLE",
            "drift_score": round(score, 4),
            "trace_drift_score": round(trace_score, 4),
            "duration_drift_score": round(duration_score, 4),
            "duration_drift_score_raw": round(duration_score_raw, 4),
            "drift_metric": DRIFT_METRIC,
            "detection_mode": DETECTION_MODE,
            "detection_threshold": DRIFT_THRESHOLD,
            "drift_type_simulated": DRIFT_TYPE if INJECT_DRIFT else "None",
            "analysis": {
                "baseline_count": len(baseline_cases),
                "current_count": len(current_cases),
                "top_k": TOP_K_TRACES,
                "top_baseline_process_freq": top_baseline_freq,
                "top_current_process_freq": top_current_freq,
                "top_baseline_process_count": top_baseline_count,
                "top_current_process_count": top_current_count
            }
        }
        return report


def evaluate_detection(
    df_events,
    drift_type,
    drift_present,
    drift_start_time,
    window_size=500,
    step_size=250,
    threshold=DRIFT_THRESHOLD,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    cases = df_events.groupby(COL_CASE_ID).agg({
        COL_ACTIVITY: lambda x: ' -> '.join(map(str, x)),
        COL_TIMESTAMP: ['min', 'max'],
    }).reset_index()
    cases.columns = ['CaseID', 'Trace', 'StartTime', 'EndTime']
    cases['Duration'] = (cases['EndTime'] - cases['StartTime']).dt.total_seconds() / 60.0
    cases = cases.sort_values('EndTime').reset_index(drop=True)

    drift_start_case_index = None
    if drift_present and drift_start_time is not None:
        drift_candidates = cases[cases['EndTime'] >= drift_start_time]
        if not drift_candidates.empty:
            drift_start_case_index = int(drift_candidates.index[0])

    all_traces = sorted(cases['Trace'].unique())
    trace_map = {t: i for i, t in enumerate(all_traces)}

    def get_vec(slice_series):
        counts = slice_series.value_counts(normalize=True)
        vec = np.zeros(len(all_traces))
        for t, freq in counts.items():
            vec[trace_map[t]] = freq
        return vec

    w_size = window_size
    step = step_size

    baseline = get_vec(cases['Trace'].iloc[0:w_size])

    detected_at = None
    scores = []
    labels = []
    indices = []
    tp = fp = fn = tn = 0

    for start in range(0, len(cases) - w_size + 1, step):
        end = start + w_size
        current = get_vec(cases['Trace'].iloc[start:end])
        trace_score = compute_distribution_distance(baseline, current)
        duration_score, _ = compute_duration_drift(
            cases.iloc[0:w_size],
            cases.iloc[start:end]
        )
        score = combine_drift_scores(trace_score, duration_score)
        scores.append(score)
        indices.append(end)

        predicted_drift = score > threshold
        window_contains_drift = (
            drift_present
            and drift_start_case_index is not None
            and end > drift_start_case_index
        )
        labels.append(1 if window_contains_drift else 0)

        if predicted_drift and window_contains_drift:
            tp += 1
        elif predicted_drift and not window_contains_drift:
            fp += 1
        elif not predicted_drift and window_contains_drift:
            fn += 1
        else:
            tn += 1

        if predicted_drift and detected_at is None:
            detected_at = end

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    detection_delay = None
    if drift_present and detected_at is not None and drift_start_case_index is not None:
        detection_delay = detected_at - drift_start_case_index

    # 自动阈值：基于 F1 最大化
    best_threshold = threshold
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    if scores:
        unique_scores = sorted(set(scores))
        if any(labels):
            for t in unique_scores:
                p = [1 if s > t else 0 for s in scores]
                tp_t = sum(1 for i in range(len(p)) if p[i] == 1 and labels[i] == 1)
                fp_t = sum(1 for i in range(len(p)) if p[i] == 1 and labels[i] == 0)
                fn_t = sum(1 for i in range(len(p)) if p[i] == 0 and labels[i] == 1)
                precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
                recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
                f1_t = (2 * precision_t * recall_t / (precision_t + recall_t)) if (precision_t + recall_t) else 0.0
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_threshold = t
                    best_precision = precision_t
                    best_recall = recall_t
        else:
            # 没有真实漂移时，选更保守阈值以减少误报
            best_threshold = max(scores)
            best_f1 = 0.0

    eval_report = {
        "detection_threshold": threshold,
        "window_size": w_size,
        "step_size": step,
        "drift_present": drift_present,
        "drift_type": drift_type,
        "drift_start_time": str(drift_start_time) if drift_start_time is not None else None,
        "drift_start_case_index": drift_start_case_index,
        "detected_at": detected_at,
        "detection_delay_cases": detection_delay,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "scores_max": round(float(np.max(scores)) if scores else 0.0, 4),
        "scores_mean": round(float(np.mean(scores)) if scores else 0.0, 4),
        "scores_std": round(float(np.std(scores)) if scores else 0.0, 4),
        "window_count": len(indices),
        "best_threshold": round(float(best_threshold), 4),
        "best_f1": round(float(best_f1), 4),
        "best_precision": round(float(best_precision), 4),
        "best_recall": round(float(best_recall), 4),
    }

    return eval_report


def generate_llm_prompt(report):
    """5. 生成给 LLM 的最终 Prompt"""
    prompt = f"""
    [System Analysis Report]
    Data Source: Help Desk Logs (Simulated Drift)
    Detection Model: {DRIFT_METRIC.upper()} Distance Analysis ({DETECTION_MODE.upper()})

    --------------------------------------------------
    Status: {report['status']}
    Drift Score: {report['drift_score']} (Threshold: {report.get('detection_threshold', DRIFT_THRESHOLD)})
    --------------------------------------------------

    [Process Changes Observed]
    1. Baseline Top Process (Top {report['analysis']['top_k']}):
    {json.dumps(report['analysis']['top_baseline_process_freq'], indent=2)}

    2. Current (Drifted) Top Process (Top {report['analysis']['top_k']}):
    {json.dumps(report['analysis']['top_current_process_freq'], indent=2)}

    [Task]
    请输出一份清晰、可读性高的 Markdown 报告，使用以下固定结构标题：
    - 总览
    - 关键变化（对比 Baseline vs Current）
    - 根因推断
    - 改进建议

    要求：
    1) 每部分使用短段落或项目符号。
    2) 必须引用数据中的具体数值（drift_score、top_k 频率/计数）。
    3) 结论尽量量化，避免空泛描述。
    """
    return prompt


# ================= 主执行逻辑 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift detection pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation and save metrics")
    parser.add_argument("--eval-window", type=int, default=500, help="Evaluation window size")
    parser.add_argument("--eval-step", type=int, default=250, help="Evaluation step size")
    parser.add_argument("--eval-threshold", type=float, default=DRIFT_THRESHOLD, help="Evaluation threshold")
    parser.add_argument("--auto-threshold", action="store_true", help="Auto-tune threshold from evaluation")
    parser.add_argument("--eval-seed", type=int, default=42, help="Random seed for evaluation")
    args = parser.parse_args()

    # 初始化
    pipeline = PipelineProcessor(FILE_PATH)

    if pipeline.step1_load_and_preprocess():
        random.seed(DRIFT_SEED)
        np.random.seed(DRIFT_SEED)

        # 数据切分：按 Case 结束时间切分，避免拆分同一 Case
        case_end = (
            pipeline.df.groupby(COL_CASE_ID)[COL_TIMESTAMP]
            .max()
            .reset_index()
            .sort_values(COL_TIMESTAMP)
            .reset_index(drop=True)
        )
        mid_case = int(len(case_end) * 0.5)
        baseline_case_ids = set(case_end.iloc[:mid_case][COL_CASE_ID])
        current_case_ids = set(case_end.iloc[mid_case:][COL_CASE_ID])

        df_baseline_raw = pipeline.df[pipeline.df[COL_CASE_ID].isin(baseline_case_ids)]
        df_current_raw = pipeline.df[pipeline.df[COL_CASE_ID].isin(current_case_ids)]

        # 1. 生成基准 Trace
        cases_baseline = pipeline.step2_generate_traces(df_baseline_raw)

        # 2. 对测试集注入漂移
        df_current_injected = pipeline.step3_inject_drift(df_current_raw)

        # 3. 生成测试 Trace (基于注入后的数据)
        cases_current = pipeline.step2_generate_traces(df_current_injected)

        # 用于评估的全量事件（包含注入后的后半段）
        df_eval = pd.concat([df_baseline_raw, df_current_injected], ignore_index=True)

        # 4. 检测与报告
        final_report = pipeline.step4_detect_and_report(cases_baseline, cases_current)

        # 5. 输出 Prompt (给人类直接复制用的)
        print("\n" + "=" * 30 + " LLM PROMPT " + "=" * 30)
        llm_input = generate_llm_prompt(final_report)
        print(llm_input)

        with open("final_llm_input_prompt.txt", "w", encoding="utf-8") as f:
            f.write(llm_input)

        # =========== 【新增】保存 JSON 给 Azure 脚本用 ===========
        json_output_path = os.path.join(os.path.dirname(__file__), "datasets", "final_report_for_azure.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ JSON Data saved for Azure script: {json_output_path}")

        if args.evaluate:
            drift_start_time = None
            if INJECT_DRIFT:
                drift_start_time = case_end[COL_TIMESTAMP].iloc[mid_case] if len(case_end) else None
            eval_report = evaluate_detection(
                df_eval,
                drift_type=DRIFT_TYPE if INJECT_DRIFT else "None",
                drift_present=INJECT_DRIFT,
                drift_start_time=drift_start_time,
                window_size=args.eval_window,
                step_size=args.eval_step,
                threshold=args.eval_threshold,
                seed=args.eval_seed,
            )
            eval_output_path = os.path.join(os.path.dirname(__file__), "datasets", "evaluation_report.json")
            with open(eval_output_path, "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Evaluation report saved: {eval_output_path}")

            if args.auto_threshold and "best_threshold" in eval_report:
                auto_th = eval_report["best_threshold"]
                final_report["detection_threshold"] = auto_th
                final_report["detection_threshold_source"] = "auto"
                final_report["status"] = "DRIFT DETECTED" if final_report["drift_score"] > auto_th else "STABLE"
                print(f"✅ Auto threshold applied: {auto_th}")

            # 同步写入主报告，方便 LLM 读取
            final_report["evaluation"] = eval_report
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            print("✅ Evaluation metrics appended to final_report_for_azure.json")

        print("\n✅ Pipeline Finished.")
