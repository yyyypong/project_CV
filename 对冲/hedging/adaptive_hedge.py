"""
自适应对冲频率算法模块
实现了基于市场状态和资产波动率动态调整对冲频率的策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pricing import BlackScholes, FiniteDifference
from models.local_vol import LocalVolatility
from hedging.greeks import GreeksCalculator
from config import (
    DEFAULT_HEDGE_FREQ, MIN_HEDGE_FREQ, MAX_HEDGE_FREQ,
    GAMMA_THRESHOLD, VEGA_THRESHOLD, RISK_FREE_RATE
)


class AdaptiveHedgeStrategy:
    """自适应对冲频率策略"""
    
    def __init__(self, risk_free_rate=None):
        """
        初始化自适应对冲策略
        
        参数：
            risk_free_rate (float): 无风险利率，默认使用配置文件中的值
        """
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
        self.greeks_calculator = GreeksCalculator(risk_free_rate=self.risk_free_rate)
        self.reset()
    
    def reset(self):
        """重置策略状态"""
        self.position = 0.0  # 标的资产头寸
        self.cash = 0.0  # 现金头寸
        self.hedge_history = []  # 对冲历史
        self.pnl_history = []  # 盈亏历史
        self.hedge_cost = 0.0  # 对冲成本
        self.total_transactions = 0  # 交易次数
    
    def calculate_hedge_frequency(self, S, K, T, sigma, vol_change_rate=0.0, option_type='call',
                                 market_impact=0.0005, transaction_cost=0.0002):
        """
        计算最优对冲频率
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            vol_change_rate (float): 波动率变化率（每天）
            option_type (str): 期权类型，'call'或'put'
            market_impact (float): 市场冲击成本系数
            transaction_cost (float): 交易成本系数
            
        返回：
            float: 最优对冲频率（每天）
        """
        # 计算Gamma和Vega
        greeks = self.greeks_calculator.all_greeks(S, K, T, sigma, option_type)
        gamma = greeks['gamma']
        vega = greeks['vega']
        
        # 计算gamma的绝对大小
        gamma_abs = abs(gamma * S**2)
        
        # 基于Gamma计算对冲频率
        if gamma_abs < GAMMA_THRESHOLD:
            # 如果Gamma很小，降低对冲频率
            gamma_freq = MIN_HEDGE_FREQ
        else:
            # Gamma较大时增加对冲频率，使用平方根规则
            gamma_freq = min(MAX_HEDGE_FREQ, np.sqrt(gamma_abs / GAMMA_THRESHOLD) * DEFAULT_HEDGE_FREQ)
        
        # 基于Vega和波动率变化率计算Vega对冲频率
        vega_abs = abs(vega)
        vol_impact = abs(vol_change_rate) * vega_abs
        
        if vol_impact < VEGA_THRESHOLD or abs(vol_change_rate) < 0.01:
            # 如果Vega很小或波动率变化很小，不需要频繁对冲
            vega_freq = MIN_HEDGE_FREQ
        else:
            # Vega较大且波动率变化显著时增加对冲频率
            vega_freq = min(MAX_HEDGE_FREQ, np.sqrt(vol_impact / VEGA_THRESHOLD) * DEFAULT_HEDGE_FREQ)
        
        # 结合Gamma和Vega对冲频率
        freq = max(gamma_freq, vega_freq)
        
        # 考虑交易成本和市场冲击
        total_cost = transaction_cost + market_impact
        
        # 使用平方根规则调整频率（平衡对冲误差和交易成本）
        optimal_freq = min(MAX_HEDGE_FREQ, freq / np.sqrt(2 * total_cost))
        
        return max(MIN_HEDGE_FREQ, optimal_freq)
    
    def delta_hedge(self, S, K, T, sigma, option_type='call', quantity=1.0, transaction_cost=0.0002):
        """
        执行一次Delta对冲
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            quantity (float): 期权头寸数量
            transaction_cost (float): 交易成本系数
            
        返回：
            tuple: (delta, new_position, transaction_cost)
        """
        # 计算Delta
        delta = self.greeks_calculator.delta(S, K, T, sigma, option_type)
        
        # 计算目标头寸
        target_position = -delta * quantity
        
        # 计算需要调整的头寸
        delta_position = target_position - self.position
        
        if abs(delta_position) > 0:
            # 计算交易成本
            cost = abs(delta_position) * S * transaction_cost
            self.hedge_cost += cost
            self.total_transactions += 1
            
            # 更新头寸
            self.position = target_position
            
            # 更新现金（考虑交易成本）
            self.cash -= delta_position * S + cost
            
        return (delta, target_position, cost if abs(delta_position) > 0 else 0.0)
    
    def run_simulation(self, S0, K, T, sigma, option_type='call', quantity=1.0, 
                      days=30, paths=1, dt=1/252, transaction_cost=0.0002, 
                      adaptive=True, fixed_freq=1.0, seed=None):
        """
        运行对冲模拟
        
        参数：
            S0 (float): 初始标的资产价格
            K (float): 期权行权价格
            T (float): 期权初始到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            quantity (float): 期权头寸数量
            days (int): 模拟天数
            paths (int): 路径数量
            dt (float): 时间步长（天）
            transaction_cost (float): 交易成本系数
            adaptive (bool): 是否使用自适应频率
            fixed_freq (float): 固定对冲频率（如果adaptive=False）
            seed (int): 随机数种子
            
        返回：
            dict: 模拟结果
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 重置策略状态
        self.reset()
        
        # 初始化
        steps_per_day = int(1 / dt)
        total_steps = days * steps_per_day
        sqrt_dt = np.sqrt(dt)
        
        # 初始化价格路径
        S = np.zeros((paths, total_steps + 1))
        S[:, 0] = S0
        
        # 生成价格路径
        for i in range(total_steps):
            Z = np.random.normal(0, 1, paths)
            S[:, i+1] = S[:, i] * np.exp((self.risk_free_rate - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z)
        
        # 初始化结果存储
        all_results = []
        
        # 对每个路径进行模拟
        for p in range(paths):
            self.reset()
            path_S = S[p]
            
            # 计算期权初始价格和Delta
            initial_price = BlackScholes(option_type=option_type, risk_free_rate=self.risk_free_rate).price(
                S0, K, T, sigma
            )
            
            # 初始现金 = 期权价格（如果做多期权则为负，做空则为正）
            self.cash = initial_price * quantity
            
            # 初始对冲
            self.delta_hedge(S0, K, T, sigma, option_type, quantity, transaction_cost)
            
            # 记录对冲历史
            self.hedge_history.append({
                'time': 0,
                'S': S0,
                'T_remaining': T,
                'position': self.position,
                'cash': self.cash,
                'hedge_freq': DEFAULT_HEDGE_FREQ if adaptive else fixed_freq
            })
            
            # 记录每天的波动率变化率（用于自适应对冲）
            vol_changes = []
            
            # 运行模拟
            for day in range(days):
                # 当天的波动率变化率（模拟值，实际应该从市场数据中获取）
                if day > 0 and len(vol_changes) > 0:
                    vol_change = np.mean(vol_changes[-min(5, len(vol_changes)):])
                else:
                    vol_change = 0.0
                
                # 计算对冲频率
                if adaptive:
                    # 当前价格和剩余时间
                    current_S = path_S[day * steps_per_day]
                    remaining_T = T - day * dt * steps_per_day
                    
                    # 自适应对冲频率
                    hedge_freq = self.calculate_hedge_frequency(
                        current_S, K, remaining_T, sigma, vol_change, option_type, transaction_cost=transaction_cost
                    )
                else:
                    hedge_freq = fixed_freq
                
                # 计算当天对冲次数
                hedge_times = max(1, int(hedge_freq))
                
                # 每天内执行对冲
                for h in range(hedge_times):
                    step = day * steps_per_day + h * (steps_per_day // hedge_times)
                    if step >= total_steps:
                        break
                    
                    current_S = path_S[step]
                    remaining_T = T - step * dt
                    
                    # 执行对冲
                    self.delta_hedge(current_S, K, remaining_T, sigma, option_type, quantity, transaction_cost)
                    
                    # 记录对冲历史
                    self.hedge_history.append({
                        'time': step * dt,
                        'S': current_S,
                        'T_remaining': remaining_T,
                        'position': self.position,
                        'cash': self.cash,
                        'hedge_freq': hedge_freq
                    })
                
                # 计算当天的隐含波动率变化（实际应该从市场数据获取）
                # 这里简化为随机波动
                daily_vol_change = np.random.normal(0, 0.01)
                vol_changes.append(daily_vol_change)
                
                # 更新波动率（这里简化为加上变化量）
                sigma = max(0.05, sigma + daily_vol_change)
            
            # 最终结算
            final_S = path_S[-1]
            final_T = T - total_steps * dt
            
            # 计算期权最终价值
            if option_type == 'call':
                option_value = max(0, final_S - K) * quantity
            else:  # put
                option_value = max(0, K - final_S) * quantity
            
            # 计算最终头寸价值
            position_value = self.position * final_S
            
            # 计算最终盈亏
            final_pnl = self.cash + position_value - option_value
            
            # 记录结果
            result = {
                'path': p,
                'final_pnl': final_pnl,
                'hedge_cost': self.hedge_cost,
                'total_transactions': self.total_transactions,
                'hedge_history': pd.DataFrame(self.hedge_history),
                'adaptive': adaptive,
                'hedge_freq': 'adaptive' if adaptive else fixed_freq
            }
            
            all_results.append(result)
        
        return all_results
    
    def compare_strategies(self, S0, K, T, sigma, option_type='call', quantity=1.0, 
                         days=30, paths=10, transaction_cost=0.0002, seed=None):
        """
        比较自适应和固定频率对冲策略
        
        参数：
            S0 (float): 初始标的资产价格
            K (float): 期权行权价格
            T (float): 期权初始到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            quantity (float): 期权头寸数量
            days (int): 模拟天数
            paths (int): 路径数量
            transaction_cost (float): 交易成本系数
            seed (int): 随机数种子
            
        返回：
            tuple: (自适应结果, 固定频率结果)
        """
        # 固定随机种子以便比较
        if seed is not None:
            np.random.seed(seed)
        
        # 运行自适应策略
        adaptive_results = self.run_simulation(
            S0, K, T, sigma, option_type, quantity, days, paths, 
            transaction_cost=transaction_cost, adaptive=True, seed=seed
        )
        
        # 运行固定频率策略（每天一次）
        fixed_results = self.run_simulation(
            S0, K, T, sigma, option_type, quantity, days, paths, 
            transaction_cost=transaction_cost, adaptive=False, fixed_freq=DEFAULT_HEDGE_FREQ, seed=seed
        )
        
        # 计算平均结果
        adaptive_pnl = np.mean([r['final_pnl'] for r in adaptive_results])
        adaptive_cost = np.mean([r['hedge_cost'] for r in adaptive_results])
        adaptive_trans = np.mean([r['total_transactions'] for r in adaptive_results])
        
        fixed_pnl = np.mean([r['final_pnl'] for r in fixed_results])
        fixed_cost = np.mean([r['hedge_cost'] for r in fixed_results])
        fixed_trans = np.mean([r['total_transactions'] for r in fixed_results])
        
        # 计算成本节省
        cost_saving = (fixed_cost - adaptive_cost) / fixed_cost * 100 if fixed_cost > 0 else 0
        
        print(f"自适应对冲策略平均盈亏: {adaptive_pnl:.2f}")
        print(f"自适应对冲策略平均成本: {adaptive_cost:.2f}")
        print(f"自适应对冲策略平均交易次数: {adaptive_trans:.2f}")
        print(f"固定频率策略平均盈亏: {fixed_pnl:.2f}")
        print(f"固定频率策略平均成本: {fixed_cost:.2f}")
        print(f"固定频率策略平均交易次数: {fixed_trans:.2f}")
        print(f"对冲成本节省: {cost_saving:.2f}%")
        
        return adaptive_results, fixed_results
    
    def plot_comparison(self, adaptive_results, fixed_results, path_idx=0):
        """
        绘制策略比较图
        
        参数：
            adaptive_results (list): 自适应策略结果
            fixed_results (list): 固定频率策略结果
            path_idx (int): 要绘制的路径索引
            
        返回：
            tuple: (fig, axes)
        """
        # 获取特定路径的结果
        adaptive = adaptive_results[path_idx]
        fixed = fixed_results[path_idx]
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 提取数据
        adaptive_history = adaptive['hedge_history']
        fixed_history = fixed['hedge_history']
        
        # 图1：价格路径
        axes[0, 0].plot(adaptive_history['time'], adaptive_history['S'], 'b-', label='资产价格')
        axes[0, 0].set_title('资产价格路径')
        axes[0, 0].set_xlabel('时间 (年)')
        axes[0, 0].set_ylabel('价格')
        axes[0, 0].grid(True)
        
        # 图2：对冲头寸
        axes[0, 1].plot(adaptive_history['time'], adaptive_history['position'], 'g-', label='自适应策略')
        axes[0, 1].plot(fixed_history['time'], fixed_history['position'], 'r--', label='固定频率策略')
        axes[0, 1].set_title('对冲头寸')
        axes[0, 1].set_xlabel('时间 (年)')
        axes[0, 1].set_ylabel('头寸')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 图3：对冲频率
        axes[1, 0].plot(adaptive_history['time'], adaptive_history['hedge_freq'], 'g-', label='自适应频率')
        axes[1, 0].axhline(y=DEFAULT_HEDGE_FREQ, color='r', linestyle='--', label='固定频率')
        axes[1, 0].set_title('对冲频率')
        axes[1, 0].set_xlabel('时间 (年)')
        axes[1, 0].set_ylabel('频率 (每天)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 图4：对冲成本累计
        cash_diff_adaptive = np.diff(np.insert(adaptive_history['cash'].values, 0, 0))
        cash_diff_fixed = np.diff(np.insert(fixed_history['cash'].values, 0, 0))
        
        cum_cost_adaptive = np.cumsum(np.abs(cash_diff_adaptive))
        cum_cost_fixed = np.cumsum(np.abs(cash_diff_fixed))
        
        axes[1, 1].plot(adaptive_history['time'], np.insert(cum_cost_adaptive, 0, 0), 'g-', label='自适应策略成本')
        axes[1, 1].plot(fixed_history['time'], np.insert(cum_cost_fixed, 0, 0), 'r--', label='固定频率策略成本')
        axes[1, 1].set_title('累计对冲成本')
        axes[1, 1].set_xlabel('时间 (年)')
        axes[1, 1].set_ylabel('成本')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 添加总结信息
        cost_saving = (fixed['hedge_cost'] - adaptive['hedge_cost']) / fixed['hedge_cost'] * 100
        fig.suptitle(f'对冲策略比较\n'
                    f'自适应策略成本: {adaptive["hedge_cost"]:.2f}, 交易次数: {adaptive["total_transactions"]}\n'
                    f'固定频率策略成本: {fixed["hedge_cost"]:.2f}, 交易次数: {fixed["total_transactions"]}\n'
                    f'成本节省: {cost_saving:.2f}%', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, axes
    
    def plot_pnl_distribution(self, adaptive_results, fixed_results):
        """
        绘制盈亏分布图
        
        参数：
            adaptive_results (list): 自适应策略结果
            fixed_results (list): 固定频率策略结果
            
        返回：
            tuple: (fig, ax)
        """
        # 提取盈亏数据
        adaptive_pnl = [r['final_pnl'] for r in adaptive_results]
        fixed_pnl = [r['final_pnl'] for r in fixed_results]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制盈亏分布
        ax.hist(adaptive_pnl, alpha=0.5, bins=20, label='自适应策略')
        ax.hist(fixed_pnl, alpha=0.5, bins=20, label='固定频率策略')
        
        # 添加均值线
        ax.axvline(np.mean(adaptive_pnl), color='g', linestyle='--', label='自适应均值')
        ax.axvline(np.mean(fixed_pnl), color='r', linestyle='--', label='固定频率均值')
        
        # 设置标题和标签
        ax.set_title('对冲策略盈亏分布')
        ax.set_xlabel('盈亏')
        ax.set_ylabel('频率')
        ax.legend()
        ax.grid(True)
        
        return fig, ax 