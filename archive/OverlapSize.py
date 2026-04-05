import matplotlib

# 【关键修复】强制使用 TkAgg 独立窗口显示，解决 PyCharm 报错
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import wasserstein_distance

# 设置中文字体 (Windows 用户通常使用 Microsoft YaHei)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据 (自动路径逻辑)
# ==========================================
current_folder = os.path.dirname(os.path.abspath(__file__))
# 尝试自动拼接路径
file_path = os.path.join(current_folder, 'synthetic_insurance_data.csv')

print(f"正在读取文件: {file_path}")
try:
    df = pd.read_csv(file_path)
    print("读取成功！")
except FileNotFoundError:
    # 备用方案：如果你把代码放在了 datasets 子文件夹里
    try:
        file_path = os.path.join(os.path.dirname(current_folder), 'synthetic_insurance_data.csv')
        df = pd.read_csv(file_path)
        print(f"在上一级目录读取成功: {file_path}")
    except:
        print("错误：文件未找到，请确认 synthetic_insurance_data.csv 的位置")
        exit()

target_col = 'Premium_Amount'
window_size = 1000  # 窗口大小


# ==========================================
# 2. 定义漂移计算函数
# ==========================================
def compute_drift(data, col, w_size, overlap):
    stride = int(w_size * (1 - overlap))
    if stride < 1: stride = 1
    ref = data[col].iloc[0:w_size]
    scores, idxs = [], []
    for start in range(0, len(data) - w_size + 1, stride):
        end = start + w_size
        curr = data[col].iloc[start:end]
        scores.append(wasserstein_distance(ref, curr))
        idxs.append(end)
    return idxs, scores


# ==========================================
# 3. 制造验证数据 (压力测试)
# ==========================================
print("正在构建验证场景...")

# 场景 A: 突变 (Sudden Drift)
df_sudden = df.copy()
drift_start = 5000
shift_val = df[target_col].std() * 1.5
df_sudden.loc[drift_start:, target_col] += shift_val

# 场景 B: 渐变 (Gradual Drift)
df_gradual = df.copy()
linear_trend = np.linspace(0, shift_val * 2, len(df) - drift_start)
df_gradual.loc[drift_start:, target_col] += linear_trend

# ==========================================
# 4. 核心逻辑：生成数据证明表格
# ==========================================
print("正在计算各策略的性能指标...")

strategies = [
    (0.0, 'Overlap 0 (无重叠)'),
    (0.33, 'Overlap 1/3'),
    (0.5, 'Overlap 1/2 (推荐)')
]

threshold = 20  # 报警阈值
results = []


def get_first_detection_delay(indices, scores, start_idx, thresh):
    for i, score in zip(indices, scores):
        if i >= start_idx and score > thresh:
            return i - start_idx
    return 99999  # 未检测到


for overlap, label in strategies:
    # 1. 计算突变场景
    idx_s, score_s = compute_drift(df_sudden, target_col, window_size, overlap)
    delay_s = get_first_detection_delay(idx_s, score_s, drift_start, threshold)

    # 2. 计算渐变场景
    idx_g, score_g = compute_drift(df_gradual, target_col, window_size, overlap)
    delay_g = get_first_detection_delay(idx_g, score_g, drift_start, threshold)

    results.append({
        '策略': label,
        '突变检测延迟 (行数)': delay_s,
        '渐变检测延迟 (行数)': delay_g
    })

# 将结果转换为表格
result_df = pd.DataFrame(results)

# ==========================================
# 5. 绘图 1：显微镜模式 (Micro View)
# ==========================================
print("正在生成显微镜细节图...")
idxs_0, scores_0 = compute_drift(df_sudden, target_col, window_size, 0.0)
idxs_50, scores_50 = compute_drift(df_sudden, target_col, window_size, 0.5)

plt.figure(1, figsize=(12, 6))
plt.plot(idxs_0, scores_0, 'o-', color='blue', label='Overlap 0 (无重叠)', markersize=8, alpha=0.5)
plt.plot(idxs_50, scores_50, 'x-', color='green', label='Overlap 1/2 (推荐)', markersize=8, alpha=1.0)
plt.axvline(drift_start, color='red', linestyle='-', linewidth=2, label='真实漂移发生时刻')
plt.xlim(4000, 7000)
plt.title('【显微镜证明】 漂移发生时的响应对比 (Index 4000-7000)')
plt.xlabel('数据索引')
plt.ylabel('漂移分数')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()

# ==========================================
# 6. 最终：打印铁证数据
# ==========================================
print("\n" + "=" * 60)
print(f"【数据证明报告】 漂移发生点: {drift_start}, 报警阈值: {threshold}")
print("我们计算了不同策略发现漂移所需的'滞后行数' (越小越好)")
print("-" * 60)
print(result_df.to_string(index=False))
print("-" * 60)

# 计算提升百分比
delay_0 = result_df.loc[0, '突变检测延迟 (行数)']
delay_50 = result_df.loc[2, '突变检测延迟 (行数)']
improvement = delay_0 - delay_50

print(f"\n🚀 核心结论: 在突变场景下，Overlap 1/2 比无重叠策略")
print(f"   【快了 {improvement} 条】数据发现问题！")
print(f"   (无重叠延迟 {delay_0} 条 vs 1/2重叠延迟 {delay_50} 条)")
print("=" * 60 + "\n")

plt.show()