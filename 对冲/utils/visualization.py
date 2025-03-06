"""
可视化工具模块
提供绘制波动率曲面、对冲绩效、希腊字母等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
import sys
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Visualizer:
    """可视化工具类"""
    
    @staticmethod
    def plot_vol_surface(vol_data, title="隐含波动率曲面", cmap='viridis', alpha=0.7):
        """
        绘制波动率曲面
        
        参数：
            vol_data (DataFrame): 波动率数据，包含列：'strike', 'time_to_maturity', 'implied_vol'
            title (str): 图表标题
            cmap (str): 色图
            alpha (float): 透明度
            
        返回：
            tuple: (fig, ax)
        """
        # 创建网格
        strikes = np.sort(vol_data['strike'].unique())
        maturities = np.sort(vol_data['time_to_maturity'].unique())
        
        # 创建空的波动率网格
        vol_grid = np.zeros((len(strikes), len(maturities)))
        
        # 填充网格
        for i, k in enumerate(strikes):
            for j, t in enumerate(maturities):
                subset = vol_data[(vol_data['strike'] == k) & (vol_data['time_to_maturity'] == t)]
                if len(subset) > 0:
                    vol_grid[i, j] = subset['implied_vol'].values[0]
        
        # 使用meshgrid创建坐标网格
        K, T = np.meshgrid(strikes, maturities)
        
        # 创建图像
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(K, T, vol_grid.T, cmap=cmap, alpha=alpha, antialiased=True)
        
        # 添加标签
        ax.set_xlabel('行权价')
        ax.set_ylabel('到期时间（年）')
        ax.set_zlabel('隐含波动率')
        ax.set_title(title)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig, ax
    
    @staticmethod
    def plot_vol_smile(vol_data, selected_maturities=None, title="波动率微笑曲线"):
        """
        绘制波动率微笑曲线
        
        参数：
            vol_data (DataFrame): 波动率数据，包含列：'strike', 'time_to_maturity', 'implied_vol'
            selected_maturities (list): 要绘制的到期时间列表，如果为None则选择几个代表性的值
            title (str): 图表标题
            
        返回：
            tuple: (fig, ax)
        """
        # 获取所有到期时间
        all_maturities = np.sort(vol_data['time_to_maturity'].unique())
        
        # 如果未指定，选择几个代表性的到期时间
        if selected_maturities is None:
            if len(all_maturities) <= 4:
                selected_maturities = all_maturities
            else:
                # 选择最短、最长和中间的到期时间
                selected_maturities = [all_maturities[0], 
                                      all_maturities[len(all_maturities) // 3],
                                      all_maturities[2 * len(all_maturities) // 3], 
                                      all_maturities[-1]]
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 对每个选定的到期时间绘制曲线
        for maturity in selected_maturities:
            # 筛选数据
            subset = vol_data[vol_data['time_to_maturity'] == maturity]
            # 按行权价排序
            subset = subset.sort_values('strike')
            
            # 绘制曲线
            ax.plot(subset['strike'], subset['implied_vol'], 'o-', label=f'到期时间: {maturity:.2f}年')
        
        # 添加标签
        ax.set_xlabel('行权价')
        ax.set_ylabel('隐含波动率')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    @staticmethod
    def plot_vol_term_structure(vol_data, selected_strikes=None, title="波动率期限结构"):
        """
        绘制波动率期限结构
        
        参数：
            vol_data (DataFrame): 波动率数据，包含列：'strike', 'time_to_maturity', 'implied_vol'
            selected_strikes (list): 要绘制的行权价列表，如果为None则选择几个代表性的值
            title (str): 图表标题
            
        返回：
            tuple: (fig, ax)
        """
        # 获取所有行权价
        all_strikes = np.sort(vol_data['strike'].unique())
        
        # 如果未指定，选择几个代表性的行权价
        if selected_strikes is None:
            if len(all_strikes) <= 4:
                selected_strikes = all_strikes
            else:
                # 选择最低、最高和中间的行权价
                selected_strikes = [all_strikes[0], 
                                   all_strikes[len(all_strikes) // 3],
                                   all_strikes[2 * len(all_strikes) // 3], 
                                   all_strikes[-1]]
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 对每个选定的行权价绘制曲线
        for strike in selected_strikes:
            # 筛选数据
            subset = vol_data[vol_data['strike'] == strike]
            # 按到期时间排序
            subset = subset.sort_values('time_to_maturity')
            
            # 绘制曲线
            ax.plot(subset['time_to_maturity'], subset['implied_vol'], 'o-', label=f'行权价: {strike:.2f}')
        
        # 添加标签
        ax.set_xlabel('到期时间（年）')
        ax.set_ylabel('隐含波动率')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    @staticmethod
    def plot_greeks_surface(strike_grid, time_grid, greek_grid, greek_name='Delta', cmap='coolwarm', alpha=0.7):
        """
        绘制希腊字母曲面
        
        参数：
            strike_grid (ndarray): 行权价网格
            time_grid (ndarray): 到期时间网格
            greek_grid (ndarray): 希腊字母值网格
            greek_name (str): 希腊字母名称
            cmap (str): 色图
            alpha (float): 透明度
            
        返回：
            tuple: (fig, ax)
        """
        # 使用meshgrid创建坐标网格
        K, T = np.meshgrid(strike_grid, time_grid)
        
        # 创建图像
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(K, T, greek_grid.T, cmap=cmap, alpha=alpha, antialiased=True)
        
        # 添加标签
        ax.set_xlabel('行权价')
        ax.set_ylabel('到期时间（年）')
        ax.set_zlabel(greek_name)
        ax.set_title(f'{greek_name}曲面')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig, ax
    
    @staticmethod
    def plot_all_greeks(greeks_data, S0, title="期权希腊字母"):
        """
        绘制所有希腊字母
        
        参数：
            greeks_data (DataFrame): 希腊字母数据，包含列：'strike', 'time_to_maturity', 'delta', 'gamma', 'vega', 'theta'
            S0 (float): 当前标的资产价格
            title (str): 图表标题
            
        返回：
            tuple: (fig, axes)
        """
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制Delta
        greeks_data.plot(x='strike', y='delta', ax=axes[0, 0])
        axes[0, 0].axvline(x=S0, color='r', linestyle='--', label='当前价格')
        axes[0, 0].set_xlabel('行权价')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Delta与行权价关系')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # 绘制Gamma
        greeks_data.plot(x='strike', y='gamma', ax=axes[0, 1])
        axes[0, 1].axvline(x=S0, color='r', linestyle='--', label='当前价格')
        axes[0, 1].set_xlabel('行权价')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Gamma与行权价关系')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 绘制Vega
        greeks_data.plot(x='strike', y='vega', ax=axes[1, 0])
        axes[1, 0].axvline(x=S0, color='r', linestyle='--', label='当前价格')
        axes[1, 0].set_xlabel('行权价')
        axes[1, 0].set_ylabel('Vega')
        axes[1, 0].set_title('Vega与行权价关系')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # 绘制Theta
        greeks_data.plot(x='strike', y='theta', ax=axes[1, 1])
        axes[1, 1].axvline(x=S0, color='r', linestyle='--', label='当前价格')
        axes[1, 1].set_xlabel('行权价')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].set_title('Theta与行权价关系')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axes
    
    @staticmethod
    def plot_hedge_performance(hedge_history, title="对冲策略绩效"):
        """
        绘制对冲策略绩效
        
        参数：
            hedge_history (DataFrame): 对冲历史数据
            title (str): 图表标题
            
        返回：
            tuple: (fig, axes)
        """
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制资产价格和对冲头寸
        axes[0, 0].plot(hedge_history['time'], hedge_history['S'], 'b-', label='资产价格')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('价格')
        axes[0, 0].set_title('标的资产价格')
        axes[0, 0].grid(True)
        
        # 创建第二个y轴
        ax2 = axes[0, 0].twinx()
        ax2.plot(hedge_history['time'], hedge_history['position'], 'g--', label='对冲头寸')
        ax2.set_ylabel('头寸')
        
        # 添加两个图例
        lines1, labels1 = axes[0, 0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 绘制现金余额
        axes[0, 1].plot(hedge_history['time'], hedge_history['cash'], 'r-')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('现金')
        axes[0, 1].set_title('现金余额')
        axes[0, 1].grid(True)
        
        # 计算每次交易的现金变化
        cash_diff = hedge_history['cash'].diff().fillna(0)
        
        # 绘制交易成本
        axes[1, 0].bar(hedge_history['time'], abs(cash_diff), color='orange')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('成本')
        axes[1, 0].set_title('交易成本')
        axes[1, 0].grid(True)
        
        # 计算累计成本
        cumulative_cost = abs(cash_diff).cumsum()
        
        # 绘制累计成本
        axes[1, 1].plot(hedge_history['time'], cumulative_cost, 'm-')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('累计成本')
        axes[1, 1].set_title('累计交易成本')
        axes[1, 1].grid(True)
        
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axes
    
    @staticmethod
    def plot_portfolio_greeks(portfolio_greeks_history, title="投资组合希腊字母随时间变化"):
        """
        绘制投资组合希腊字母随时间变化
        
        参数：
            portfolio_greeks_history (DataFrame): 投资组合希腊字母历史数据
            title (str): 图表标题
            
        返回：
            tuple: (fig, axes)
        """
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制Delta
        axes[0, 0].plot(portfolio_greeks_history['time'], portfolio_greeks_history['delta'], 'b-')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('投资组合Delta')
        axes[0, 0].grid(True)
        
        # 绘制Gamma
        axes[0, 1].plot(portfolio_greeks_history['time'], portfolio_greeks_history['gamma'], 'g-')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('投资组合Gamma')
        axes[0, 1].grid(True)
        
        # 绘制Vega
        axes[1, 0].plot(portfolio_greeks_history['time'], portfolio_greeks_history['vega'], 'c-')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('Vega')
        axes[1, 0].set_title('投资组合Vega')
        axes[1, 0].grid(True)
        
        # 绘制Theta
        axes[1, 1].plot(portfolio_greeks_history['time'], portfolio_greeks_history['theta'], 'm-')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].set_title('投资组合Theta')
        axes[1, 1].grid(True)
        
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axes
    
    @staticmethod
    def plot_vol_predictions(actual_vols, predicted_vols, dates, title="波动率预测与实际值对比"):
        """
        绘制波动率预测与实际值对比
        
        参数：
            actual_vols (ndarray): 实际波动率
            predicted_vols (ndarray): 预测波动率
            dates (ndarray): 对应的日期
            title (str): 图表标题
            
        返回：
            tuple: (fig, ax)
        """
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制实际值和预测值
        ax.plot(dates, actual_vols, 'b-', label='实际波动率')
        ax.plot(dates, predicted_vols, 'r--', label='预测波动率')
        
        # 添加标签
        ax.set_xlabel('日期')
        ax.set_ylabel('波动率')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # 旋转日期标签以避免重叠
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        return fig, ax
    
    @staticmethod
    def plot_vol_prediction_error(actual_vols, predicted_vols, dates, title="波动率预测误差"):
        """
        绘制波动率预测误差
        
        参数：
            actual_vols (ndarray): 实际波动率
            predicted_vols (ndarray): 预测波动率
            dates (ndarray): 对应的日期
            title (str): 图表标题
            
        返回：
            tuple: (fig, axes)
        """
        # 计算预测误差
        error = predicted_vols - actual_vols
        abs_error = np.abs(error)
        rel_error = abs_error / actual_vols
        
        # 创建图像
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制绝对误差
        axes[0].bar(dates, abs_error, color='orange')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('绝对误差')
        axes[0].set_title('波动率预测绝对误差')
        axes[0].grid(True)
        
        # 绘制相对误差
        axes[1].bar(dates, rel_error * 100, color='purple')
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('相对误差 (%)')
        axes[1].set_title('波动率预测相对误差 (%)')
        axes[1].grid(True)
        
        # 旋转日期标签以避免重叠
        for ax in axes:
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axes
    
    @staticmethod
    def plot_strategy_comparison(adaptive_results, fixed_results, title="对冲策略比较"):
        """
        比较不同对冲策略的绩效
        
        参数：
            adaptive_results (list): 自适应策略结果
            fixed_results (list): 固定频率策略结果
            title (str): 图表标题
            
        返回：
            tuple: (fig, axes)
        """
        # 提取绩效指标
        adaptive_pnl = [r['final_pnl'] for r in adaptive_results]
        adaptive_cost = [r['hedge_cost'] for r in adaptive_results]
        adaptive_trans = [r['total_transactions'] for r in adaptive_results]
        
        fixed_pnl = [r['final_pnl'] for r in fixed_results]
        fixed_cost = [r['hedge_cost'] for r in fixed_results]
        fixed_trans = [r['total_transactions'] for r in fixed_results]
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制盈亏对比
        axes[0].boxplot([adaptive_pnl, fixed_pnl], labels=['自适应策略', '固定频率策略'])
        axes[0].set_ylabel('盈亏')
        axes[0].set_title('盈亏对比')
        axes[0].grid(True)
        
        # 绘制对冲成本对比
        axes[1].boxplot([adaptive_cost, fixed_cost], labels=['自适应策略', '固定频率策略'])
        axes[1].set_ylabel('对冲成本')
        axes[1].set_title('对冲成本对比')
        axes[1].grid(True)
        
        # 绘制交易次数对比
        axes[2].boxplot([adaptive_trans, fixed_trans], labels=['自适应策略', '固定频率策略'])
        axes[2].set_ylabel('交易次数')
        axes[2].set_title('交易次数对比')
        axes[2].grid(True)
        
        # 调整布局
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        return fig, axes
    
    @staticmethod
    def plot_correlation_matrix(data, title="特征相关性矩阵"):
        """
        绘制特征相关性矩阵
        
        参数：
            data (DataFrame): 数据
            title (str): 图表标题
            
        返回：
            tuple: (fig, ax)
        """
        # 计算相关矩阵
        corr = data.corr()
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建热图
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, ax=ax, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        
        # 设置标题
        ax.set_title(title)
        
        # 调整布局
        plt.tight_layout()
        
        return fig, ax 