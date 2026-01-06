import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_search_r1_reversed_paper_style():
    file_path = 'search-r1.xlsx'
    
    # ================== 1. 样式设置 (顶级会议风格) ==================
    # 全局设置，确保所有元素字号足够大
    plt.rcParams.update({
        'font.family': 'serif',         # 使用衬线体 (类似 Times New Roman)
        'font.size': 14,                # 基础字号
        'axes.titlesize': 18,           # 标题大小
        'axes.labelsize': 20,           # xy轴标签大小 (加大)
        'xtick.labelsize': 16,          # x轴刻度大小
        'ytick.labelsize': 16,          # y轴刻度大小
        'legend.fontsize': 13,          # 图例大小
        'lines.linewidth': 2.5,         # 线条默认宽度
        'axes.linewidth': 1.5,          # 坐标轴边框宽度
        'xtick.major.width': 1.5,       # 刻度线宽度
        'ytick.major.width': 1.5,
    })
    # ==============================================================

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # --- 2. 数据处理 (Step <= 450) ---
    max_step = 450
    df = df[df.iloc[:, 0] <= max_step]
    steps = df.iloc[:, 0]
    
    # 假设第一列列名为 "Step"，如果不是，这里可以手动指定 label
    x_label_name = df.columns[0] 

    # === F1 数据 ===
    f1_coll_data = df.iloc[:, 3] 
    f1_succ_data = df.iloc[:, 4] 
    f1_all = df.iloc[:, [3, 4]]
    f1_mean = f1_all.mean(axis=1)
    f1_std = f1_all.std(axis=1)

    # === EM 数据 ===
    em_coll_data = df.iloc[:, 5] 
    em_succ_data = df.iloc[:, 6] 
    em_all = df.iloc[:, [5, 6]]
    em_mean = em_all.mean(axis=1)
    em_std = em_all.std(axis=1)

    # --- 3. 绘图 ---
    plt.figure(figsize=(10, 6)) # 保持宽高比，具体可根据论文栏宽调整 (例如 (8, 5))

    # === Search-r1-F1 (蓝色) ===
    # 均值 (实线, 最粗, 层级最高)
    plt.plot(steps, f1_mean, color='#1f77b4', linewidth=3, label='Search-r1-F1 (Mean)', zorder=5)
    # Successful (虚线)
    plt.plot(steps, f1_succ_data, color='#1f77b4', linestyle='--', alpha=0.6, linewidth=1.5, label='Search-r1-F1 (successful)', zorder=4)
    # Collapsed (点线)
    plt.plot(steps, f1_coll_data, color='#1f77b4', linestyle=':', alpha=0.6, linewidth=1.5, label='Search-r1-F1 (collapsed)', zorder=4)
    # 阴影
    f1_lower = (f1_mean - f1_std).clip(lower=0)
    plt.fill_between(steps, f1_lower, f1_mean + f1_std, color='#1f77b4', alpha=0.15, edgecolor='none', zorder=1)


    # === Search-r1-EM (橙色) ===
    # 均值 (实线, 最粗)
    plt.plot(steps, em_mean, color='#ff7f0e', linewidth=3, label='Search-r1-EM (Mean)', zorder=5)
    # Successful (虚线)
    plt.plot(steps, em_succ_data, color='#ff7f0e', linestyle='--', alpha=0.6, linewidth=1.5, label='Search-r1-EM (successful)', zorder=4)
    # Collapsed (点线)
    plt.plot(steps, em_coll_data, color='#ff7f0e', linestyle=':', alpha=0.6, linewidth=1.5, label='Search-r1-EM (collapsed)', zorder=4)
    # 阴影
    em_lower = (em_mean - em_std).clip(lower=0)
    plt.fill_between(steps, em_lower, em_mean + em_std, color='#ff7f0e', alpha=0.15, edgecolor='none', zorder=1)

    # --- 4. 装饰 ---
    plt.xlabel(x_label_name, fontweight='bold') # 坐标轴加粗
    plt.ylabel('Score', fontweight='bold')
    
    plt.xticks([100, 200, 300, 400])
    plt.xlim(0, 450)
    
    # 改进的图例：去掉边框，位置左上
    plt.legend(loc='upper left', frameon=False, ncol=1)
    
    # 网格线稍微淡一点
    plt.grid(True, linestyle='--', alpha=0.4)

    output_filename = 'Search-r1_PaperStyle.pdf'
    # 保存为PDF (矢量图) 且 dpi=300 确保高清晰度
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"修正完成 (Paper Style)！\n保存为: {output_filename}")

if __name__ == "__main__":
    plot_search_r1_reversed_paper_style()