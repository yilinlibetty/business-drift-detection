import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta
from scipy.stats import wasserstein_distance


# ==========================================
# 第一步：数据模拟与升维 (Data Generation)
# 既然原始 CSV 缺时间维度，我们现场生成一个完美的 Event Log
# ==========================================
def generate_mock_event_log():
    print("[1/4] 正在生成带有时间维度的标准 Event Log...")
    activities_normal = ['Application', 'Credit Check', 'Risk Assessment', 'Notify']
    activities_drift = ['Application', 'AI Quick Scan', 'Auto Approval', 'Notify']  # 漂移后的流程：多了AI，少了人工

    data = []
    # 1. 正常时期 (2023年)
    curr_time = datetime(2023, 1, 1)
    for i in range(1000):  # 1000个案例
        case_id = f"C_2023_{i}"
        for act in activities_normal:
            data.append({'CaseID': case_id, 'Activity': act, 'Timestamp': curr_time})
            curr_time += timedelta(minutes=random.randint(10, 60))  # 模拟时间推移

    # 2. 漂移时期 (2024年)
    curr_time = datetime(2024, 1, 1)  # 时间断点
    for i in range(1000):
        case_id = f"C_2024_{i}"
        # 混合一点旧流程，模拟逐渐漂移，但主要是新流程
        path = activities_drift if random.random() > 0.2 else activities_normal
        for act in path:
            data.append({'CaseID': case_id, 'Activity': act, 'Timestamp': curr_time})
            curr_time += timedelta(minutes=random.randint(5, 30))  # AI 处理得更快

    df = pd.DataFrame(data)
    print(f"      -> 生成数据 {len(df)} 行。")
    return df


# ==========================================
# 第二步 & 第三步：Parser 解析 + 跑模型 (Detection)
# ==========================================
def run_drift_detection_pipeline(df):
    print("[2/4] 启动检测模型 (1/2 Overlap + Wasserstein Distance)...")

    # 1. Parser: 将 Event Log 转化为 Trace (路径)
    # 按 CaseID 聚合，把 Activity 拼成字符串 "A -> B -> C"
    # 同时保留该 Case 的结束时间作为时间轴
    cases = df.sort_values('Timestamp').groupby('CaseID').agg({
        'Activity': lambda x: ' -> '.join(x),
        'Timestamp': 'max'  # 取最后一步的时间作为该案例的时间
    }).reset_index()

    cases = cases.sort_values('Timestamp')  # 按时间排序

    # 2. 准备滑动窗口
    window_size = 500
    overlap = 0.5
    step = int(window_size * (1 - overlap))

    # 把 Trace 变成数字 (One-Hot 频率向量) 才能算距离
    all_traces = cases['Activity'].unique()
    trace_to_idx = {t: i for i, t in enumerate(all_traces)}

    def get_dist_vector(slice_df):
        counts = slice_df['Activity'].value_counts(normalize=True)
        vec = np.zeros(len(all_traces))
        for trace, freq in counts.items():
            vec[trace_to_idx[trace]] = freq
        return vec

    # 3. 跑模型 (滑动扫描)
    # 设定基准 (前500个作为 Learning Phase)
    baseline_window = cases.iloc[0:window_size]
    baseline_vec = get_dist_vector(baseline_window)

    drift_report = None

    for start in range(0, len(cases) - window_size + 1, step):
        end = start + window_size
        current_window = cases.iloc[start:end]
        current_vec = get_dist_vector(current_window)

        # 计算距离 (模型的核心)
        score = wasserstein_distance(baseline_vec, current_vec)

        # 设定阈值 (这里简化演示，实际可用 3-Sigma)
        if score > 0.1:  # 假设 0.1 是敏感线
            print(f"🔴 [Drift Alert] 在窗口 {start}-{end} 检测到漂移! 分数: {score:.4f}")

            # ==========================================
            # 第四步：生成案卷 (JSON Output)
            # 既然抓到了，就提取证据发给 LLM
            # ==========================================
            print("[3/4] 正在提取证据并生成 JSON...")

            # 统计 Top 5 路径
            def get_top_traces(cdf):
                return cdf['Activity'].value_counts(normalize=True).head(5).to_dict()

            drift_report = {
                "report_meta": {
                    "detection_model": "Sliding Window (Overlap 1/2) + Wasserstein",
                    "drift_score": round(score, 4),
                    "timestamp": str(current_window['Timestamp'].min())
                },
                "process_comparison": {
                    "baseline_state (2023 Normal)": get_top_traces(baseline_window),
                    "current_state (2024 Drifted)": get_top_traces(current_window)
                },
                "mermaid_graph": generate_mermaid(get_top_traces(baseline_window), get_top_traces(current_window))
            }
            break  # 抓到一个就停止，演示用

    return drift_report


def generate_mermaid(stats_before, stats_after):
    """辅助函数：生成 Mermaid 流程图代码"""
    code = ["graph LR"]

    def add_edges(stats, prefix):
        edges = {}
        for trace, freq in stats.items():
            steps = trace.split(' -> ')
            for i in range(len(steps) - 1):
                key = (f"{prefix}_{steps[i].replace(' ', '_')}", f"{prefix}_{steps[i + 1].replace(' ', '_')}")
                edges[key] = edges.get(key, 0) + freq
        return edges

    # Baseline Subgraph
    code.append("subgraph Baseline [2023 标准流程]")
    for (src, tgt), w in add_edges(stats_before, "Old").items():
        code.append(f'{src} -- "{w * 100:.0f}%" --> {tgt}')
    code.append("end")

    # Drifted Subgraph
    code.append("subgraph Current [2024 漂移流程]")
    for (src, tgt), w in add_edges(stats_after, "New").items():
        code.append(f'{src} -- "{w * 100:.0f}%" --> {tgt}')
    code.append("end")

    return "\n".join(code)


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据
    df_event_log = generate_mock_event_log()

    # 2. 跑全流程
    report = run_drift_detection_pipeline(df_event_log)

    # 3. 保存结果
    if report:
        with open('final_llm_input.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("[4/4] ✅ 任务完成！文件已生成: final_llm_input.json")
        print("下一步：请将此 JSON 文件的内容复制到 Dify 的 Prompt 中。")
    else:
        print("未检测到漂移，请调整阈值。")