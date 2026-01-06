import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_m_ask_smooth_paper_style():
    file_path = 'sft_vs_nosft.xlsx'
    
    # ================== 1. 样式设置 (保持完全一致) ==================
    # 这里的设置与上一张图完全相同，确保放在论文里看起来是一套图
    plt.rcParams.update({
        'font.family': 'serif',         # 衬线体
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

    # === 参数设置 ===
    window_size = 10  # 保持你的平滑窗口设置
    
    # 2. 检查文件
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # --- 3. 数据过滤 (Step <= 700) ---
    max_step = 700
    df = df[df.iloc[:, 0] <= max_step]
    steps = df.iloc[:, 0]
    
    # 获取X轴名称 (假设第一列)
    x_label_name = df.columns[0]

    # --- 4. 数据平滑处理函数 (保持原逻辑) ---
    def get_smooth_stats(data_cols):
        raw_mean = data_cols.mean(axis=1)
        raw_std = data_cols.std(axis=1)
        # 滑动平均
        smooth_mean = raw_mean.rolling(window=window_size, min_periods=1).mean()
        smooth_std = raw_std.rolling(window=window_size, min_periods=1).mean()
        return smooth_mean, smooth_std

    # === 第一组: M-ASK w/o SFT (索引 3, 4) ===
    wo_sft_data = df.iloc[:, [3, 4]]
    wo_mean, wo_std = get_smooth_stats(wo_sft_data)

    # === 第二组: M-ASK w SFT (索引 5, 6) ===
    w_sft_data = df.iloc[:, [5, 6]]
    w_mean, w_std = get_smooth_stats(w_sft_data)

    # --- 5. 绘图 ---
    plt.figure(figsize=(10, 6))

    # === 画 M-ASK w/o SFT (蓝色) ===
    # 线宽设为 3，zorder 设为 5 (确保线在阴影上面)
    plt.plot(steps, wo_mean, 
             label='M-ASK w/o SFT', color='#1f77b4', linewidth=3, zorder=5)
    
    # 计算阴影
    wo_lower = (wo_mean - wo_std).clip(lower=0)
    wo_upper = wo_mean + wo_std
    # alpha 设为 0.15，zorder 设为 1 (在最底层)
    plt.fill_between(steps, wo_lower, wo_upper, 
                     color='#1f77b4', alpha=0.15, edgecolor='none', zorder=1)

    # === 画 M-ASK w SFT (橙色) ===
    plt.plot(steps, w_mean, 
             label='M-ASK w SFT', color='#ff7f0e', linewidth=3, zorder=5)
    
    # 计算阴影
    w_lower = (w_mean - w_std).clip(lower=0)
    w_upper = w_mean + w_std
    plt.fill_between(steps, w_lower, w_upper, 
                     color='#ff7f0e', alpha=0.15, edgecolor='none', zorder=1)

    # --- 6. 图表装饰 (统一风格) ---
    plt.xlabel(x_label_name, fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xlim(0, 700)
    
    # 去掉图例边框 (frameon=False)，更简洁
    plt.legend(loc='upper left', frameon=False, ncol=1)
    
    # 网格线设置
    plt.grid(True, linestyle='--', alpha=0.4)

    # --- 7. 保存 ---
    output_filename = 'M-ASK_PaperStyle.pdf'
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"处理完成 (Paper Style)！\n文件保存为: {output_filename}")

if __name__ == "__main__":
    plot_m_ask_smooth_paper_style()