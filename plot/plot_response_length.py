import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_response_length_compare_paper_style():
    # ================== 1. 样式设置 (顶级会议统一风格) ==================
    # 保持与前两张图完全一致的配置
    plt.rcParams.update({
        'font.family': 'serif',         # 衬线体
        'font.size': 14,                # 基础字号
        'axes.titlesize': 18,           # 标题大小
        'axes.labelsize': 20,           # xy轴标签大小
        'xtick.labelsize': 16,          # x轴刻度大小
        'ytick.labelsize': 16,          # y轴刻度大小
        'legend.fontsize': 13,          # 图例大小
        'lines.linewidth': 2.5,         # 线条默认宽度
        'axes.linewidth': 1.5,          # 坐标轴边框宽度
        'xtick.major.width': 1.5,       # 刻度线宽度
        'ytick.major.width': 1.5,
    })
    # ==============================================================

    # === 文件路径设置 ===
    file_search_r1 = 'search-r1_res_len.xlsx'
    file_m_ask = 'm-ask_res_len.xlsx'
    
    # === 参数设置 ===
    window_size = 10
    max_step = 1000  # 截取前 1000 step
    
    # --- 智能读取函数 (保留原逻辑) ---
    def read_data_file(file_path):
        """尝试多种方式读取文件（Excel -> CSV UTF-8 -> CSV GBK）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        # 1. 尝试作为 Excel 读取
        try:
            return pd.read_excel(file_path)
        except Exception:
            pass 

        # 2. 尝试作为 CSV (UTF-8) 读取
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception:
            pass

        # 3. 尝试作为 CSV (GBK/GB18030) 读取
        try:
            return pd.read_csv(file_path, encoding='gb18030')
        except Exception:
            pass
            
        raise ValueError(f"无法读取文件 {file_path}，请检查格式。")

    # --- 读取数据 ---
    try:
        print(f"正在读取: {file_search_r1} ...")
        df_search = read_data_file(file_search_r1)
        
        print(f"正在读取: {file_m_ask} ...")
        df_mask = read_data_file(file_m_ask)
    except Exception as e:
        print(f"\n读取文件失败: {e}")
        return

    # --- 数据处理与平滑函数 ---
    def get_smooth_data(df):
        # 过滤 Step <= max_step
        df_subset = df[df.iloc[:, 0] <= max_step].copy()
        
        steps = df_subset.iloc[:, 0]
        # 取第2列作为数据列
        raw_data = df_subset.iloc[:, 1]
        
        # 滑动平均
        smooth_data = raw_data.rolling(window=window_size, min_periods=1).mean()
        return steps, smooth_data

    # 处理数据
    steps_search, val_search = get_smooth_data(df_search)
    steps_mask, val_mask = get_smooth_data(df_mask)

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))

    # === 画 M-ASK (蓝色) ===
    # 增加 linewidth 到 3，zorder 设为 5
    plt.plot(steps_mask, val_mask, 
             label='M-ASK', color='#1f77b4', linewidth=3, zorder=5)
    
    # === 画 Search-r1 (橙色) ===
    # 增加 linewidth 到 3，zorder 5
    plt.plot(steps_search, val_search, 
             label='Search-r1', color='#ff7f0e', linewidth=3, zorder=5)

    # --- 图表装饰 ---
    plt.xlabel('Step', fontweight='bold')
    plt.ylabel('Response Length', fontweight='bold')
    plt.xlim(0, max_step)
    
    # 样式统一：去掉边框 (frameon=False)，位置自动最佳 (loc='best')
    plt.legend(loc='best', frameon=False)
    
    # 网格线淡化
    plt.grid(True, linestyle='--', alpha=0.4)

    # --- 保存 ---
    output_filename = 'Response_Length_Comparison_PaperStyle.pdf'
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n处理完成 (Paper Style)！\n文件已保存为: {output_filename}")

if __name__ == "__main__":
    plot_response_length_compare_paper_style()