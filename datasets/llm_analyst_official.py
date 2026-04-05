import json
import os
from openai import OpenAI

# ================= ⚙️ 配置区域 =================

# 1. 您的 OpenAI API Key (sk-proj-...)
# ⚠️ 注意：不要上传或泄露您的 Key
API_KEY = "sk-40hwexwUh2GuW3jaUiMNoUARdEDd1CxZnQgr2I3VyD84soWg" # 请在本地运行时填入，不要上传到GitLab
# 2. 模型名称
MODEL_NAME = "gpt-5.2"  # 或 gpt-4o, gpt-3.5-turbo

# 3. Base URL (官方默认)
BASE_URL = "http://35.220.164.252:3888/v1"
# 4. 文件路径 (自动对接 Pipeline)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 读取 run_full_pipeline.py 生成的 JSON
INPUT_FILE = os.path.join(BASE_DIR, "final_report_for_azure.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "examples", "Final_Drift_Analysis_Report.md")


# ===========================================
TOTAL_STEPS = 3


def print_progress(step, message):
    width = 24
    filled = int(width * step / TOTAL_STEPS)
    bar = "#" * filled + "-" * (width - filled)
    print(f"Progress: [{bar}] {step}/{TOTAL_STEPS} {message}")

def load_report_data(filepath):
    print_progress(1, f"读取数据: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")
        print("   -> 请先运行 run_full_pipeline.py 生成数据！")
        return None


def run_analysis_official(json_data):
    print_progress(2, f"正在呼叫 OpenAI ({MODEL_NAME})")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 构建 Prompt
    data_str = json.dumps(json_data, indent=2, ensure_ascii=False)

    system_prompt = (
        "你是一位资深的 BPM 业务流程挖掘专家。\n"
        "请根据用户提供的 JSON 漂移检测数据，撰写一份清晰、可读性高的 Markdown 分析报告。\n\n"
        "硬性要求：\n"
        "1) 使用固定结构标题：\n"
        "   - 总览\n"
        "   - 关键变化（对比 Baseline vs Current）\n"
        "   - 根因推断\n"
        "   - 改进建议\n"
        "2) 每个部分使用短段落或项目符号，避免长句堆叠。\n"
        "3) 必须引用 JSON 里的具体数据（例如 drift_score、top_k 频率/计数）。\n"
        "4) 结论尽量量化，避免空泛描述。\n"
        "5) 用中文输出，格式为 Markdown。\n"
    )

    user_prompt = (
        "以下是系统检测到的漂移数据（JSON）。\n"
        "请严格按要求输出报告：\n"
        "```json\n"
        f"{data_str}\n"
        "```"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # 确保安装了 httpx: pip install httpx
    data = load_report_data(INPUT_FILE)
    if data:
        report = run_analysis_official(data)
        if report:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(report)
            print_progress(3, f"报告生成成功: {OUTPUT_FILE}")
