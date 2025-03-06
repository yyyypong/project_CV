"""
期权动态对冲策略示例
展示如何使用局部波动率模型和自适应对冲策略进行期权对冲
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from models.pricing import BlackScholes
from models.local_vol import LocalVolatility
from models.vol_surface import VolatilitySurface
from hedging.greeks import GreeksCalculator
from hedging.adaptive_hedge import AdaptiveHedgeStrategy
from utils.data_loader import DataLoader, DataProcessor
from utils.visualization import Visualizer


def run_hedging_example():
    """运行期权对冲示例"""
    print("期权动态对冲策略示例")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../results/hedging_example')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 设置期权参数
    print("\n1. 设置期权参数")
    print("-" * 40)
    
    S0 = 100.0  # 初始标的资产价格
    K = 100.0   # 平值期权行权价格
    T = 0.5     # 半年期权（到期时间）
    sigma = 0.2  # 波动率
    
    # 创建期权定价模型
    bs_model = BlackScholes(option_type='call')
    
    # 计算期权初始价格
    option_price = bs_model.price(S0, K, T, sigma)
    print(f"期权初始价格: {option_price:.4f}")
    
    # 2. 对冲策略设置
    print("\n2. 对冲策略设置")
    print("-" * 40)
    
    # 创建希腊字母计算器
    greeks_calculator = GreeksCalculator()
    
    # 计算初始希腊字母
    greeks = greeks_calculator.all_greeks(S0, K, T, sigma)
    print("初始希腊字母:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.6f}")
    
    # 3. 模拟标的资产价格路径
    print("\n3. 模拟标的资产价格路径")
    print("-" * 40)
    
    # 设置模拟参数
    days = 30  # 模拟30天
    dt = 1/252  # 每日时间步长
    steps_per_day = 1
    total_steps = days * steps_per_day
    
    # 设置随机种子以便重现
    np.random.seed(42)
    
    # 模拟价格路径
    S_path = np.zeros(total_steps + 1)
    S_path[0] = S0
    
    for i in range(total_steps):
        # 生成随机增量
        dW = np.random.normal(0, 1)
        
        # 更新价格（几何布朗运动）
        dS = S_path[i] * (0.05 * dt + sigma * np.sqrt(dt) * dW)
        S_path[i+1] = S_path[i] + dS
    
    # 4. 执行不同对冲策略
    print("\n4. 执行不同对冲策略")
    print("-" * 40)
    
    # 创建对冲策略
    hedge_strategy = AdaptiveHedgeStrategy()
    
    # 比较自适应和固定频率策略
    print("比较自适应和固定频率策略...")
    adaptive_results, fixed_results = hedge_strategy.compare_strategies(
        S0, K, T, sigma, days=days, paths=10, seed=42
    )
    
    # 5. 分析对冲结果
    print("\n5. 分析对冲结果")
    print("-" * 40)
    
    # 计算平均结果
    adaptive_pnl = np.mean([r['final_pnl'] for r in adaptive_results])
    adaptive_cost = np.mean([r['hedge_cost'] for r in adaptive_results])
    adaptive_trans = np.mean([r['total_transactions'] for r in adaptive_results])
    
    fixed_pnl = np.mean([r['final_pnl'] for r in fixed_results])
    fixed_cost = np.mean([r['hedge_cost'] for r in fixed_results])
    fixed_trans = np.mean([r['total_transactions'] for r in fixed_results])
    
    # 计算成本节省
    cost_saving = (fixed_cost - adaptive_cost) / fixed_cost * 100 if fixed_cost > 0 else 0
    
    print(f"自适应对冲策略平均盈亏: {adaptive_pnl:.4f}")
    print(f"自适应对冲策略平均成本: {adaptive_cost:.4f}")
    print(f"自适应对冲策略平均交易次数: {adaptive_trans:.1f}")
    print(f"固定频率策略平均盈亏: {fixed_pnl:.4f}")
    print(f"固定频率策略平均成本: {fixed_cost:.4f}")
    print(f"固定频率策略平均交易次数: {fixed_trans:.1f}")
    print(f"对冲成本节省: {cost_saving:.2f}%")
    
    # 6. 可视化结果
    print("\n6. 可视化结果")
    print("-" * 40)
    
    # 绘制对冲策略比较
    print("绘制对冲策略比较...")
    fig, axes = hedge_strategy.plot_comparison(adaptive_results, fixed_results)
    plt.savefig(os.path.join(results_dir, 'hedge_strategy_comparison.png'))
    plt.close(fig)
    
    # 绘制盈亏分布
    fig, ax = hedge_strategy.plot_pnl_distribution(adaptive_results, fixed_results)
    plt.savefig(os.path.join(results_dir, 'pnl_distribution.png'))
    plt.close(fig)
    
    # 绘制单次对冲过程详情（使用第一次模拟的结果）
    fig, axes = hedge_strategy.plot_comparison(adaptive_results, fixed_results, path_idx=0)
    plt.savefig(os.path.join(results_dir, 'single_hedge_process.png'))
    plt.close(fig)
    
    # 7. 额外实验：不同波动率环境下的对冲效果
    print("\n7. 额外实验：不同波动率环境下的对冲效果")
    print("-" * 40)
    
    # 测试不同波动率水平
    volatility_levels = [0.1, 0.2, 0.3, 0.4]
    adaptive_costs = []
    fixed_costs = []
    
    for vol in volatility_levels:
        print(f"测试波动率 = {vol}...")
        
        # 运行对冲模拟
        a_results, f_results = hedge_strategy.compare_strategies(
            S0, K, T, vol, days=days, paths=5, seed=42
        )
        
        # 记录平均对冲成本
        a_cost = np.mean([r['hedge_cost'] for r in a_results])
        f_cost = np.mean([r['hedge_cost'] for r in f_results])
        
        adaptive_costs.append(a_cost)
        fixed_costs.append(f_cost)
        
        # 计算成本节省
        saving = (f_cost - a_cost) / f_cost * 100 if f_cost > 0 else 0
        print(f"波动率 {vol}: 自适应成本 {a_cost:.4f}, 固定成本 {f_cost:.4f}, 节省 {saving:.2f}%")
    
    # 绘制不同波动率下的对冲成本
    plt.figure(figsize=(10, 6))
    plt.plot(volatility_levels, adaptive_costs, 'g-o', label='自适应策略')
    plt.plot(volatility_levels, fixed_costs, 'r-s', label='固定频率策略')
    plt.xlabel('波动率')
    plt.ylabel('对冲成本')
    plt.title('不同波动率环境下的对冲成本')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'hedging_cost_vs_volatility.png'))
    plt.close()
    
    # 计算成本节省
    cost_savings = [(f - a) / f * 100 if f > 0 else 0 for f, a in zip(fixed_costs, adaptive_costs)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(volatility_levels, cost_savings)
    plt.xlabel('波动率')
    plt.ylabel('成本节省 (%)')
    plt.title('不同波动率环境下的对冲成本节省')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'cost_savings_vs_volatility.png'))
    plt.close()
    
    print("\n完成！结果已保存到:", results_dir)
    print("=" * 60)


if __name__ == "__main__":
    run_hedging_example() 