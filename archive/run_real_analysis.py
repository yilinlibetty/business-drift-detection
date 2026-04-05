import pandas as pd
import json
import numpy as np
import os
import uuid
import matplotlib

matplotlib.use('TkAgg')  # 强制弹窗
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ================= 配置区域 =================
# 你的真实文件路径
FILE_PATH = r"/datasets/finale.csv"

COL_CASE_ID = 'Case ID'
COL_ACTIVITY = 'Activity'
COL_TIMESTAMP = 'Complete Timestamp'

WINDOW_SIZE = 500
OVERLAP_RATIO = 0.5


# ===========================================

def load_real_data():
    print(f"[1/5] 正在读取文件: {FILE_PATH}")
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        exit()

    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP])
    df = df.sort_values(COL_TIMESTAMP)

    print("[2/5] 正在构建 Trace...")
    cases = df.groupby(COL_CASE_ID).agg({
        COL_ACTIVITY: lambda x: ' -> '.join(map(str, x)),
        COL_TIMESTAMP: 'max'
    }).reset_index()

    cases = cases.sort_values(COL_TIMESTAMP)
    print(f"      -> 提取出 {len(cases)} 个独立案例")
    return cases


def get_distribution(data_slice, trace_map):
    counts = data_slice.value_counts(normalize=True)
    vec = np.zeros(len(trace_map))
    for t, freq in counts.items():
        if t in trace_map:
            vec[trace_map[t]] = freq
    return vec


def get_top_traces(data_slice, top_n=5):
    return data_slice.value_counts(normalize=True).head(top_n).to_dict()


def generate_mermaid(stats_before, stats_after):
    code = ["graph LR"]

    def add_edges(stats, prefix):
        edges = {}
        for trace, freq in stats.items():
            steps = trace.split(' -> ')
            for i in range(len(steps) - 1):
                src = steps[i].replace(' ', '_').replace(':', '').replace('-', '_').replace('(', '').replace(')', '')
                tgt = steps[i + 1].replace(' ', '_').replace(':', '').replace('-', '_').replace('(', '').replace(')',
                                                                                                                 '')
                key = (f"{prefix}_{src}", f"{prefix}_{tgt}")
                edges[key] = edges.get(key, 0) + freq
        return edges

    code.append("subgraph Baseline [旧流程]")
    for (src, tgt), w in add_edges(stats_before, "Old").items():
        if w > 0.05:
            code.append(f'{src} -- "{w * 100:.0f}%" --> {tgt}')
    code.append("end")

    code.append("subgraph Current [新流程]")
    for (src, tgt), w in add_edges(stats_after, "New").items():
        if w > 0.05:
            code.append(f'{src} -- "{w * 100:.0f}%" --> {tgt}')
    code.append("end")
    return "\n".join(code)


def run_forced_visualization_pipeline(cases):
    print(f"[3/5] 启动全量扫描 (Window={WINDOW_SIZE})")

    step = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))
    all_traces = cases[COL_ACTIVITY].unique()
    trace_map = {t: i for i, t in enumerate(all_traces)}

    baseline_window = cases[COL_ACTIVITY].iloc[0:WINDOW_SIZE]
    baseline_vec = get_distribution(baseline_window, trace_map)

    scores = []
    indices = []

    # 扫描
    for start in range(0, len(cases) - WINDOW_SIZE + 1, step):
        end = start + WINDOW_SIZE
        current_window = cases[COL_ACTIVITY].iloc[start:end]
        current_vec = get_distribution(current_window, trace_map)
        score = wasserstein_distance(baseline_vec, current_vec)
        scores.append(score)
        indices.append(end)

    # 统计数据
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    max_idx_pos = np.argmax(scores)
    max_case_index = indices[max_idx_pos]

    # 设定阈值 (Mean + 2 Std)
    threshold = avg_score + 2 * std_score
    print(f"      📊 统计: Mean={avg_score:.4f}, Max={max_score:.4f}, Threshold={threshold:.4f}")

    # ================= 核心修改：强制画图 =================
    print("[4/5] 正在绘制趋势图 (无论是否报警)...")
    plt.figure(figsize=(12, 6))
    plt.plot(indices, scores, label='Drift Score', color='blue')
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.axhline(y=max_score, color='red', linestyle=':', label=f'Max Peak ({max_score:.4f})')
    plt.title('Drift Score Trend (Help Desk Log)')
    plt.xlabel('Case Index')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ================= 核心修改：保底抓取逻辑 =================
    drift_events = []

    # 策略：如果没有任何点超过阈值，就强制抓取最大值的那个点
    force_capture = False
    if max_score < threshold:
        print("⚠️ 警告: 数据非常稳定，未超过统计阈值。")
        print("🚀 启动 [高灵敏度模式]: 强制抓取波动最大的点作为漂移事件。")
        threshold = max_score * 0.99  # 降低阈值，刚好能抓到最大值
        force_capture = True

    # 提取事件
    if force_capture:
        # 只抓最大那一个
        print(f"      🚩 强制捕获峰值! 时间点: Case {max_case_index}, 分数: {max_score:.4f}")

        best_start = max_case_index - WINDOW_SIZE
        target_window = cases[COL_ACTIVITY].iloc[best_start:max_case_index]
        timestamp_str = str(cases[COL_TIMESTAMP].iloc[max_case_index - 1])

        event_report = {
            "event_id": str(uuid.uuid4()),
            "type": "Micro-Drift (Forced Capture)" if force_capture else "Significant Drift",
            "timestamp": timestamp_str,
            "case_index_end": max_case_index,
            "drift_score": round(max_score, 4),
            "process_patterns": {
                "baseline": get_top_traces(baseline_window),
                "current_drift": get_top_traces(target_window)
            },
            "mermaid_code": generate_mermaid(get_top_traces(baseline_window), get_top_traces(target_window))
        }
        drift_events.append(event_report)
        # 在图上标记
        plt.plot(max_case_index, max_score, 'ro', markersize=10, label='Forced Detection')

    else:
        # 正常抓取逻辑
        for i, score in enumerate(scores):
            if score > threshold:
                # 简单处理：只显示超过阈值的点
                plt.plot(indices[i], score, 'ro')
                # (这里为了代码简洁，如果正常抓到了，我们暂不生成重复的JSON，逻辑同上)
                # 你可以用之前的逻辑来处理多点抓取
                pass

        # 如果正常抓到了，我们也把最大那个生成报告出来
        print(f"      🚩 捕获显著漂移! 峰值: {max_score:.4f}")
        best_start = max_case_index - WINDOW_SIZE
        target_window = cases[COL_ACTIVITY].iloc[best_start:max_case_index]
        timestamp_str = str(cases[COL_TIMESTAMP].iloc[max_case_index - 1])

        event_report = {
            "event_id": str(uuid.uuid4()),
            "type": "Significant Drift",
            "timestamp": timestamp_str,
            "case_index_end": max_case_index,
            "drift_score": round(max_score, 4),
            "process_patterns": {
                "baseline": get_top_traces(baseline_window),
                "current_drift": get_top_traces(target_window)
            },
            "mermaid_code": generate_mermaid(get_top_traces(baseline_window), get_top_traces(target_window))
        }
        drift_events.append(event_report)

    # 保存报告
    if drift_events:
        output_file = 'final_helpdesk_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"dataset": "Help Desk", "events": drift_events}, f, indent=2, ensure_ascii=False)
        print(f"\n[5/5] ✅ 报告已生成: {output_file}")
        print("      请将 JSON 内容复制给 Dify/LLM。")

    print("正在显示图表...")
    plt.show()  # 这一行现在绝对会执行


if __name__ == "__main__":
    df_cases = load_real_data()
    run_forced_visualization_pipeline(df_cases)