import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from scipy.stats import wasserstein_distance


# ================= 工具函数：生成数据 =================
def generate_data(drift_type='none', total_cases=2000):
    """
    drift_type:
      - 'none': 全程正常 (测试误报率)
      - 'strong': 剧烈漂移 (测试召回率)
      - 'weak': 微弱漂移 (测试灵敏度)
    """
    activities_normal = ['A', 'B', 'C', 'D']
    activities_drift = ['A', 'X', 'Y', 'D']  # 新流程

    data = []
    drift_start_idx = 1000

    for i in range(total_cases):
        case_id = str(i)
        # 确定当前流程
        if i < drift_start_idx:
            path = activities_normal
        else:
            if drift_type == 'none':
                path = activities_normal
            elif drift_type == 'strong':
                # 80% 的概率走新流程
                path = activities_drift if random.random() < 0.8 else activities_normal
            elif drift_type == 'weak':
                # 只有 10% 的概率走新流程 (很难测!)
                path = activities_drift if random.random() < 0.1 else activities_normal

        # 简单生成时间戳
        ts = datetime(2023, 1, 1) + timedelta(minutes=i * 10)
        for act in path:
            data.append({'CaseID': case_id, 'Activity': act, 'Timestamp': ts})
            ts += timedelta(minutes=1)

    return pd.DataFrame(data), drift_start_idx


# ================= 模型核心逻辑 =================
def run_model(df, drift_start_idx):
    # 1. Trace 提取
    cases = df.groupby('CaseID')['Activity'].agg(lambda x: '->'.join(x)).reset_index()

    # 2. 向量化
    all_traces = cases['Activity'].unique()
    trace_map = {t: i for i, t in enumerate(all_traces)}

    def get_vec(slice_series):
        counts = slice_series.value_counts(normalize=True)
        vec = np.zeros(len(all_traces))
        for t, freq in counts.items():
            vec[trace_map[t]] = freq
        return vec

    # 3. 滑动窗口检测
    w_size = 500
    step = 250  # 1/2 Overlap

    baseline = get_vec(cases['Activity'].iloc[0:w_size])
    detected_at = None

    for start in range(0, len(cases) - w_size + 1, step):
        end = start + w_size
        current = get_vec(cases['Activity'].iloc[start:end])

        score = wasserstein_distance(baseline, current)

        # 阈值设定为 0.1 (根据经验)
        if score > 0.1:
            detected_at = end
            break  # 发现即停止

    return detected_at


# ================= 主测试流程 =================
print("===== 开始模型验证 (Model Verification) =====")

# 1. 测试误报率 (False Positive Test)
print("\n[测试 1] 空载测试 (无漂移数据)...")
df_clean, _ = generate_data('none')
res_clean = run_model(df_clean, 1000)
if res_clean is None:
    print("✅ 通过! 模型保持安静，无误报。")
else:
    print(f"❌ 失败! 模型在没有漂移时报警了 (位置: {res_clean})。")

# 2. 测试准确性 & 延迟 (Accuracy & Latency)
print("\n[测试 2] 标准测试 (强漂移数据)...")
df_strong, true_start = generate_data('strong')
res_strong = run_model(df_strong, true_start)
if res_strong:
    delay = res_strong - true_start
    print(f"✅ 通过! 成功检测到漂移。")
    print(f"   - 真实发生点: {true_start}")
    print(f"   - 检测报告点: {res_strong}")
    print(f"   - 延迟 (Latency): {delay} Cases")
else:
    print("❌ 失败! 模型漏掉了明显的漂移。")

# 3. 测试灵敏度 (Sensitivity)
print("\n[测试 3] 压力测试 (微弱漂移 - 仅10%混合)...")
df_weak, true_start = generate_data('weak')
res_weak = run_model(df_weak, true_start)
if res_weak:
    print(f"✅ 优秀! 即使只有 10% 的噪音，模型依然抓住了漂移。")
else:
    print("⚠️ 正常。漂移太微弱，被当做噪音忽略了 (这是符合预期的，防止过敏)。")

print("\n===== 验证结束 =====")