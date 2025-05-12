import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

df = pd.read_csv("D:/Microsoft VS Code/hs300_data.csv")

df['date'] = pd.to_datetime(df['date'])

# 按股票代码和日期排序
df = df.sort_values(['code', 'date'])

# 重新计算振幅（若需要）
# df['amplitude'] = (df['high'] / df['low'] - 1) * 100


def calculate_A_factor_vectorized(group, N=160, lambda_=0.7):
    group = group.copy()
    group['return_pct'] = group['return'] / 100
    amplitudes = group['amplitude'].values
    returns = group['return_pct'].values
    
    if len(amplitudes) < N:  # 数据不足直接返回空列
        group['A_factor'] = np.nan
        return group
    
    threshold = int(N * lambda_)
    
    # 生成滑动窗口视图（shape: [n_windows, N]）
    amp_win = swv(amplitudes, N, axis=0)
    ret_win = swv(returns, N, axis=0)
    
    # 每个窗口内排序并选择低振幅部分
    sorted_indices = np.argsort(amp_win, axis=1)
    selected_returns = np.take_along_axis(ret_win, sorted_indices[:, :threshold], axis=1)
    a_factors = selected_returns.sum(axis=1)
    
    # 对齐结果到原始数据
    result = np.full(len(group), np.nan)
    result[N-1:] = a_factors  # 第一个完整窗口结束位置为N-1
    group['A_factor'] = result
    
    return group

def calculate_B_factor_vectorized(group, N=160, lambda_=0.3):
    """
    向量化计算B因子（高振幅反转因子）
    参数:
        group: 单只股票的DataFrame
        N: 回看窗口期（默认160）
        lambda_: 高振幅切割比例（默认30%）
    返回:
        添加B_factor列的DataFrame
    """
    group = group.copy()
    group['return_pct'] = group['return'] / 100  # 收益率转为小数
    amplitudes = group['amplitude'].values
    returns = group['return_pct'].values
    
    if len(amplitudes) < N:  # 数据不足返回空值
        group['B_factor'] = np.nan
        return group
    
    threshold = int(N * (1 - lambda_))  # 高振幅起始位置（如160*0.7=112）
    
    # 生成滑动窗口视图（shape: [n_windows, N]）
    amp_win = swv(amplitudes, N, axis=0)
    ret_win = swv(returns, N, axis=0)
    
    # 每个窗口内按振幅降序排序，选择高振幅部分
    sorted_indices = np.argsort(amp_win, axis=1)  # 默认升序
    selected_returns = np.take_along_axis(ret_win, sorted_indices[:, threshold:], axis=1)
    b_factors = selected_returns.sum(axis=1)
    
    # 对齐结果到原始数据
    result = np.full(len(group), np.nan)
    result[N-1:] = b_factors  # 第一个完整窗口结束位置为N-1
    group['B_factor'] = result
    
    return group

# 按股票分组计算
df = df.groupby('code').apply(calculate_A_factor_vectorized)

# 计算下一期收益率
df = df.reset_index(drop=True)

df['next_return'] = df.groupby('code')['return_pct'].shift(-1)

# 剔除无效值
valid_df = df.dropna(subset=['A_factor', 'next_return'])

# 计算IC（Rank IC）
ic_series = valid_df.groupby('date').apply(
    lambda x: x['A_factor'].corr(x['next_return'], method='spearman'))
ic_mean = ic_series.mean()
ic_ir = ic_mean / ic_series.std()

print(f"IC均值: {ic_mean:.3f}")
print(f"ICIR: {ic_ir:.2f}")


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 创建画布
fig = plt.figure(figsize=(16, 12), dpi=100)
gs = fig.add_gridspec(2, 2)

# 子图1：IC时间序列
ax1 = fig.add_subplot(gs[0, :])
ic_series.plot(ax=ax1, color='steelblue', alpha=0.7, title='Rank IC时间序列')
ax1.axhline(ic_mean, color='tomato', linestyle='--', label=f'均值 ({ic_mean:.3f})')
ax1.axhline(0, color='gray', linestyle=':')
ax1.set_ylabel('Rank IC')
ax1.legend()

# 子图2：IC分布直方图
ax2 = fig.add_subplot(gs[1, 0])
sns.histplot(ic_series, bins=30, kde=True, ax=ax2, color='teal', alpha=0.6)
ax2.axvline(ic_mean, color='tomato', linestyle='--', label=f'均值 ({ic_mean:.3f})')
ax2.set_title('IC分布直方图')
ax2.set_xlabel('Rank IC')
ax2.legend()

# 子图3：滚动IC均值（12个月窗口）
ax3 = fig.add_subplot(gs[1, 1])
rolling_ic = ic_series.rolling(12, min_periods=6).mean()
rolling_ic.plot(ax=ax3, color='purple', alpha=0.8, title='滚动12期IC均值')
ax3.axhline(ic_mean, color='tomato', linestyle='--')
ax3.axhline(0, color='gray', linestyle=':')
ax3.set_ylabel('滚动IC均值')

# 添加全局标题
plt.suptitle(f"因子IC分析 | IC均值: {ic_mean:.3f}  ICIR: {ic_ir:.2f}", y=1.0, fontsize=14)
plt.tight_layout()
plt.show()

import akshare as ak
# 1. 获取中证500指数历史数据
def get_csi500_data():
    # 使用akshare获取中证500指数数据
    csi500_df = ak.stock_zh_index_daily(symbol="sh000905")
    csi500_df = csi500_df[['date', 'close']].copy()
    csi500_df['date'] = pd.to_datetime(csi500_df['date'])
    csi500_df.set_index('date', inplace=True)
    csi500_df.sort_index(inplace=True)
    
    # 计算日收益率
    csi500_df['return'] = csi500_df['close'].pct_change()
    return csi500_df['return']

# 获取中证500收益率数据
csi500_returns = get_csi500_data()

# 2. 你的原始回测代码
# 生成分组信号 (使用5分位数)
valid_df['factor_quantile'] = valid_df.groupby('date')['A_factor'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))

# 计算分组收益率
group_returns = valid_df.groupby(['date', 'factor_quantile'])['next_return'].mean().unstack()
group_returns.columns = [f'Q{i+1}' for i in range(5)]

# 确保日期为datetime类型并排序
group_returns.index = pd.to_datetime(group_returns.index)
group_returns.sort_index(inplace=True)

# 3. 对齐中证500数据与回测数据
# 只保留回测期间有数据的日期
csi500_aligned = csi500_returns[csi500_returns.index.isin(group_returns.index)]

# 计算累计收益
cum_returns = (1 + group_returns).cumprod()
csi500_cum = (1 + csi500_aligned).cumprod()

# 将中证500基准加入累计收益DataFrame
cum_returns['CSI500'] = csi500_cum

# 4. 计算回测指标
def calculate_performance(returns):
    total_return = returns.add(1).prod() - 1
    annual_return = (1 + total_return) ** (252/len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility
    max_drawdown = (returns.add(1).cumprod().cummax() - returns.add(1).cumprod()).max()
    return pd.Series({
        '年化收益': annual_return,
        '波动率': volatility,
        '夏普比率': sharpe,
        '最大回撤': max_drawdown,
        '胜率': (returns > 0).mean()
    })

# 将中证500基准加入性能比较
all_returns = group_returns.copy()
all_returns['CSI500'] = csi500_aligned
performance = all_returns.apply(calculate_performance).T

# 5. 可视化结果
# plt.style.use('seaborn')
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

# 累计收益曲线
cum_returns[['Q1', "Q3", 'Q5', 'CSI500']].plot(ax=ax[0], lw=2,
                                              style=['r-', 'g-', 'b-', 'k--'])
ax[0].set_title('中证500多头策略', fontsize=14)
ax[0].set_ylabel('累计收益 (倍数)', fontsize=12)
ax[0].axhline(1, color='black', linestyle=':', linewidth=1)
ax[0].legend(['Q1 (最低因子值)', 'Q3 (中位数)', 'Q5 (最高因子值)', '中证500指数'],
            loc='upper left', fontsize=10)

# 添加网格线
ax[0].grid(True, linestyle='--', alpha=0.7)

# 性能指标表格
col_labels = performance.columns.tolist()
row_labels = performance.index.tolist()
table_values = performance.round(3).values

ax[1].axis('off')
performance_table = ax[1].table(cellText=table_values,
                              colLabels=col_labels,
                              rowLabels=row_labels,
                              loc='center',
                              cellLoc='center')
performance_table.auto_set_font_size(False)
performance_table.set_fontsize(10)
performance_table.scale(1.2, 1.2)
ax[1].set_title('策略绩效指标', y=0.8, fontsize=14)

plt.tight_layout()
plt.show()