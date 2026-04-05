#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_drift_test.py
-----------------
这个脚本演示了如何在不依赖复杂第三方平台的情况下，
使用 Python 标准数据科学库 (Pandas, Scipy) 进行数据漂移检测。

主要逻辑：
1. 生成/加载两组数据：
   - 参考数据集 (Reference Data): 代表模型训练时的基准数据。
   - 当前数据集 (Current Data): 代表生产环境中的新数据。
2. 使用 Kolmogorov-Smirnov (KS) 检验比较两个数据集中特征的分布。
3. 如果 p-value 小于阈值 (如 0.05)，则认为发生了统计学上的显著漂移。

依赖库:
pip install pandas numpy scipy
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import sys


def generate_mock_data(n_samples=1000):
    """
    生成模拟数据用于测试。
    包含三个特征：
    - feature_stable: 分布保持不变
    - feature_drifted_mean: 均值发生漂移
    - feature_drifted_var: 方差发生漂移
    """
    np.random.seed(42)

    # 1. 生成参考数据 (Reference Data / Training Data)
    ref_data = pd.DataFrame({
        'feature_stable': np.random.normal(0, 1, n_samples),
        'feature_drifted_mean': np.random.normal(0, 1, n_samples),
        'feature_drifted_var': np.random.normal(0, 1, n_samples)
    })

    # 2. 生成当前数据 (Current Data / Production Data)
    # 这里的参数被故意修改以模拟漂移
    curr_data = pd.DataFrame({
        'feature_stable': np.random.normal(0, 1, n_samples),  # 不变
        'feature_drifted_mean': np.random.normal(2, 1, n_samples),  # 均值漂移 (0 -> 2)
        'feature_drifted_var': np.random.normal(0, 3, n_samples)  # 方差漂移 (1 -> 3)
    })

    return ref_data, curr_data


def run_drift_detection(reference_df, current_df, threshold=0.05):
    """
    对每一列执行 KS 检验以检测漂移。

    Args:
        reference_df (pd.DataFrame): 基准数据
        current_df (pd.DataFrame): 当前数据
        threshold (float): P-value 阈值，低于此值视为拒绝原假设（即认为分布不同）

    Returns:
        pd.DataFrame: 包含每列检测结果的报告
    """
    drift_report = []

    # 确保列名一致
    common_columns = list(set(reference_df.columns) & set(current_df.columns))

    print(f"正在分析 {len(common_columns)} 个特征...\n")

    for col in common_columns:
        # 获取两组数据中该列的值
        ref_values = reference_df[col].dropna()
        curr_values = current_df[col].dropna()

        # 执行 KS 检验 (Kolmogorov-Smirnov test)
        # statistic: KS统计量，表示两个分布累积分布函数的最大差值
        # pvalue: P值
        test_result = ks_2samp(ref_values, curr_values)

        is_drifted = test_result.pvalue < threshold

        drift_report.append({
            "Feature": col,
            "Drift Detected": "YES" if is_drifted else "No",
            "P-Value": f"{test_result.pvalue:.5f}",
            "Statistical Score": f"{test_result.statistic:.4f}",
            "Threshold": threshold
        })

    # 转换为 DataFrame 以便展示
    report_df = pd.DataFrame(drift_report)

    # 调整列顺序
    report_df = report_df[["Feature", "Drift Detected", "P-Value", "Statistical Score", "Threshold"]]
    return report_df


def main():
    parser = argparse.ArgumentParser(description="运行简单的数据漂移检测")
    parser.add_argument("--samples", type=int, default=1000, help="生成的样本数量")
    parser.add_argument("--threshold", type=float, default=0.05, help="漂移检测的 P-Value 阈值")

    args = parser.parse_args()

    print("-" * 50)
    print("步骤 1: 生成模拟数据...")
    ref_df, curr_df = generate_mock_data(n_samples=args.samples)
    print(f"参考数据集大小: {ref_df.shape}")
    print(f"当前数据集大小: {curr_df.shape}")

    print("-" * 50)
    print("步骤 2: 运行漂移算法 (KS Test)...")
    drift_results = run_drift_detection(ref_df, curr_df, threshold=args.threshold)

    print("-" * 50)
    print("步骤 3: 检测报告")
    print("-" * 50)
    print(drift_results.to_string(index=False))
    print("-" * 50)

    # 如果检测到漂移，以非零状态码退出（方便 CI/CD 管道集成）
    if "YES" in drift_results["Drift Detected"].values:
        print("警告: 检测到数据漂移！")
        # sys.exit(1) # 如果需要在管道中阻断流程，取消此行注释
    else:
        print("成功: 未检测到显著漂移。")


if __name__ == "__main__":
    main()