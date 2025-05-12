import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import akshare as ak
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# 数据加载与预处理
df = pd.read_csv("D:/Microsoft VS Code/hs300_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date'])  # 按股票和日期排序

# 重置索引，确保'code'只是列名而不是索引
df = df.reset_index(drop=True)

# 计算收益率（假设原始数据中的return字段是百分比，需要转为小数）
df['return_pct'] = df['return'] / 100

# Step1: 计算市场平均收益和σ值
df['market_return'] = df.groupby('date')['return_pct'].transform('mean')
df['sigma'] = (abs(df['return_pct'] - df['market_return']) / 
               (abs(df['return_pct']) + abs(df['market_return']) + 0.1))

# Step2 & 3: 定义ST因子计算函数（按月度分组）
def calculate_ST_factor(group, delta=0.7):
    if len(group) < 2:  # 至少需要2天数据计算协方差
        group['ST_factor'] = np.nan
        return group
    sigma = group['sigma'].values
    # 对σ降序排序，k=1表示最高凸显性
    ranks = (-sigma).argsort().argsort() + 1  # 获取排名（1-based）
    delta_k = delta ** ranks
    omega = delta_k / np.mean(delta_k)  # 归一化凸显权重
    # 计算协方差（ω与收益率）
    cov = np.cov(omega, group['return_pct'])[0, 1]
    # 仅将ST值赋给该月最后一天
    group['ST_factor'] = np.nan
    group.loc[group.index[-1], 'ST_factor'] = cov
    return group

# 按月分组计算ST因子
df['month'] = df['date'].dt.to_period('M')
df = df.groupby(['code', 'month'], group_keys=False).apply(calculate_ST_factor)

# 前向填充ST因子（确保下个月使用当月的ST值）
df['ST_factor'] = df.groupby('code')['ST_factor'].ffill()

# 计算下一期收益率（对齐到月末）
df['next_return'] = df.groupby('code')['return_pct'].shift(-1)

# 剔除无效值
valid_df = df.dropna(subset=['ST_factor', 'next_return'])

# ================== 回测分析 ==================
# 1. 计算Rank IC
ic_series = valid_df.groupby('date').apply(
    lambda x: x['ST_factor'].corr(x['next_return'], method='spearman'))
ic_mean = ic_series.mean()
ic_ir = ic_mean / ic_series.std()
print(f"ST因子IC均值: {ic_mean:.3f}")
print(f"ST因子ICIR: {ic_ir:.2f}")

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

# 2. 分组收益计算（五分位数）
valid_df['factor_quantile'] = valid_df.groupby('date')['ST_factor'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
group_returns = valid_df.groupby(['date', 'factor_quantile'])['next_return'].mean().unstack()
group_returns.columns = [f'Q{i+1}' for i in range(5)]
group_returns.index = pd.to_datetime(group_returns.index)
group_returns = group_returns.sort_index()

# 确保日期为datetime类型并排序
group_returns.index = pd.to_datetime(group_returns.index)
group_returns.sort_index(inplace=True)

# 3. 对齐中证500数据与回测数据
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