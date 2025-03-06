"""
期权波动率曲面建模与动态对冲系统主程序
实现了完整的工作流程，包括数据加载、模型构建、对冲策略执行和结果可视化
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 导入项目模块
from models.pricing import BlackScholes, MonteCarlo, FiniteDifference
from models.local_vol import LocalVolatility
from models.vol_surface import VolatilitySurface
from hedging.greeks import GreeksCalculator
from hedging.adaptive_hedge import AdaptiveHedgeStrategy
from ml.nn_vol_predict import VolatilityPredictor
from utils.data_loader import DataLoader, DataProcessor
from utils.visualization import Visualizer


def main():
    """主程序入口"""
    print("期权波动率曲面建模与动态对冲系统")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 数据加载和处理
    print("\n1. 数据加载和处理")
    print("-" * 40)
    
    # 创建数据加载器
    data_loader = DataLoader()
    
    # 加载期权数据和股票数据
    print("加载市场数据...")
    option_data = data_loader.load_option_data()
    stock_data = data_loader.load_stock_data()
    
    # 处理期权数据
    print("处理期权数据...")
    option_data = DataProcessor.process_option_data(option_data)
    
    # 计算历史波动率
    print("计算历史波动率...")
    stock_data = DataProcessor.calculate_historical_volatility(stock_data)
    
    # 获取当前标的资产价格
    S0 = 100.0  # 这里使用模拟数据的初始价格
    
    # 准备波动率曲面数据
    print("准备波动率曲面数据...")
    vol_surface_data = DataProcessor.prepare_vol_surface_data(option_data)
    
    # 2. 波动率曲面建模
    print("\n2. 波动率曲面建模")
    print("-" * 40)
    
    # 创建波动率曲面模型
    print("构建波动率曲面...")
    vol_surface = VolatilitySurface(method='cubic')
    vol_surface.calibrate(vol_surface_data, S0)
    
    # 可视化波动率曲面
    print("可视化波动率曲面...")
    fig, ax = Visualizer.plot_vol_surface(vol_surface_data)
    plt.savefig(os.path.join(results_dir, 'vol_surface.png'))
    
    # 可视化波动率微笑
    fig, ax = Visualizer.plot_vol_smile(vol_surface_data)
    plt.savefig(os.path.join(results_dir, 'vol_smile.png'))
    
    # 可视化波动率期限结构
    fig, ax = Visualizer.plot_vol_term_structure(vol_surface_data)
    plt.savefig(os.path.join(results_dir, 'vol_term_structure.png'))
    
    # 3. 局部波动率模型
    print("\n3. 局部波动率模型")
    print("-" * 40)
    
    # 创建局部波动率模型
    print("校准局部波动率模型...")
    local_vol = LocalVolatility()
    local_vol.calibrate(vol_surface_data, S0)
    
    # 模拟局部波动率模型下的价格路径
    print("模拟价格路径...")
    paths = local_vol.simulate_path(S0, 1.0, 252, num_paths=5, seed=42)
    
    # 绘制模拟路径
    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(np.linspace(0, 1, 253), paths[i])
    plt.xlabel('时间 (年)')
    plt.ylabel('价格')
    plt.title('局部波动率模型价格路径模拟')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'local_vol_paths.png'))
    
    # 4. 希腊字母计算
    print("\n4. 希腊字母计算")
    print("-" * 40)
    
    # 创建希腊字母计算器
    print("计算希腊字母...")
    greeks_calculator = GreeksCalculator()
    
    # 计算不同行权价和到期时间的希腊字母
    strikes = np.linspace(0.8 * S0, 1.2 * S0, 21)
    maturities = np.array([0.1, 0.5, 1.0])
    
    # 计算Delta表面
    delta_grid = np.zeros((len(strikes), len(maturities)))
    gamma_grid = np.zeros_like(delta_grid)
    vega_grid = np.zeros_like(delta_grid)
    theta_grid = np.zeros_like(delta_grid)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            # 使用局部波动率模型获取波动率
            sigma = local_vol.get_local_vol(S0, K, T)
            
            # 计算希腊字母
            greeks = greeks_calculator.all_greeks(S0, K, T, sigma, option_type='call')
            delta_grid[i, j] = greeks['delta']
            gamma_grid[i, j] = greeks['gamma']
            vega_grid[i, j] = greeks['vega']
            theta_grid[i, j] = greeks['theta']
    
    # 可视化希腊字母
    for greek_name, greek_grid in [('Delta', delta_grid), ('Gamma', gamma_grid), 
                                  ('Vega', vega_grid), ('Theta', theta_grid)]:
        fig, ax = Visualizer.plot_greeks_surface(strikes, maturities, greek_grid, greek_name)
        plt.savefig(os.path.join(results_dir, f'{greek_name.lower()}_surface.png'))
    
    # 5. 动态对冲策略
    print("\n5. 动态对冲策略")
    print("-" * 40)
    
    # 创建自适应对冲策略
    print("执行对冲策略模拟...")
    hedge_strategy = AdaptiveHedgeStrategy()
    
    # 模拟对冲
    K = S0  # 平值期权
    T = 0.5  # 6个月期权
    sigma = 0.2  # 波动率
    
    # 比较自适应和固定频率策略
    print("比较自适应和固定频率策略...")
    adaptive_results, fixed_results = hedge_strategy.compare_strategies(
        S0, K, T, sigma, option_type='call', days=30, paths=10, seed=42
    )
    
    # 可视化对冲策略比较
    print("可视化对冲策略比较...")
    fig, axes = hedge_strategy.plot_comparison(adaptive_results, fixed_results)
    plt.savefig(os.path.join(results_dir, 'hedge_comparison.png'))
    
    fig, ax = hedge_strategy.plot_pnl_distribution(adaptive_results, fixed_results)
    plt.savefig(os.path.join(results_dir, 'hedge_pnl_distribution.png'))
    
    # 6. 波动率预测模型
    print("\n6. 波动率预测模型")
    print("-" * 40)
    
    # 创建机器学习特征
    print("创建机器学习特征...")
    ml_data = DataProcessor.create_ml_features(option_data, stock_data)
    
    # 分割数据集
    print("分割数据集...")
    train_data, val_data, test_data = DataProcessor.split_option_data(ml_data)
    
    # 创建波动率预测模型
    print("训练波动率预测模型...")
    vol_predictor = VolatilityPredictor(model_type='mlp')
    
    # 设置特征和目标列
    feature_cols = ['moneyness', 'time_to_maturity', 'historical_vol', 'oi_change']
    target_col = 'implied_vol'
    
    # 训练模型（设置较小的轮数以便快速演示）
    vol_predictor.train(
        train_data, 
        feature_columns=feature_cols, 
        target_column=target_col,
        epochs=50,  # 减少训练轮数以便快速演示
        batch_size=32
    )
    
    # 保存模型
    vol_predictor.save_model(os.path.join(results_dir, 'vol_predictor_model'))
    
    # 可视化训练历史
    fig, axes = vol_predictor.plot_training_history()
    plt.savefig(os.path.join(results_dir, 'vol_predictor_training.png'))
    
    # 7. 波动率预测
    print("\n7. 波动率预测")
    print("-" * 40)
    
    # 使用测试集预测波动率
    print("预测波动率...")
    X_test = test_data[feature_cols]
    y_test = test_data[target_col].values.reshape(-1, 1)
    
    # 绘制预测结果
    fig, ax = vol_predictor.plot_predictions(X_test, y_test, n_samples=50)
    plt.savefig(os.path.join(results_dir, 'vol_predictions.png'))
    
    # 预测未来波动率曲面
    print("预测未来波动率曲面...")
    future_surfaces = vol_predictor.predict_vol_surface(vol_surface_data, future_days=5)
    
    # 可视化预测的波动率曲面
    for day in range(min(3, len(future_surfaces))):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        vol_predictor.plot_predicted_vol_surface(future_surfaces[day], day=day, ax=ax)
        plt.savefig(os.path.join(results_dir, f'predicted_vol_surface_day{day+1}.png'))
    
    print("\n完成！结果已保存到:", results_dir)
    print("=" * 60)


if __name__ == "__main__":
    main() 