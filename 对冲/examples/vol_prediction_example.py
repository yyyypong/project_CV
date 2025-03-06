"""
波动率曲面预测示例
展示如何使用神经网络模型预测波动率曲面和期权定价
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
from ml.nn_vol_predict import VolatilityPredictor
from utils.data_loader import DataLoader, DataProcessor
from utils.visualization import Visualizer


def run_vol_prediction_example():
    """运行波动率预测示例"""
    print("波动率曲面预测示例")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../results/vol_prediction')
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
    S0 = 100.0  # 使用模拟数据中的初始价格
    
    # 2. 创建特征数据
    print("\n2. 创建特征数据")
    print("-" * 40)
    
    # 创建机器学习特征
    print("创建机器学习特征...")
    ml_data = DataProcessor.create_ml_features(option_data, stock_data)
    
    # 打印特征统计信息
    print("\n特征数据统计信息:")
    print(ml_data.describe())
    
    # 查看相关性
    corr = ml_data[['moneyness', 'time_to_maturity', 'implied_vol', 'historical_vol', 'oi_change']].corr()
    print("\n特征相关性矩阵:")
    print(corr)
    
    # 绘制相关性矩阵
    fig, ax = Visualizer.plot_correlation_matrix(ml_data[['moneyness', 'time_to_maturity', 'implied_vol', 'historical_vol', 'oi_change']])
    plt.savefig(os.path.join(results_dir, 'feature_correlation.png'))
    plt.close(fig)
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据")
    print("-" * 40)
    
    # 分割数据集
    print("分割数据集...")
    train_data, val_data, test_data = DataProcessor.split_option_data(ml_data)
    
    # 设置特征和目标列
    feature_cols = ['moneyness', 'time_to_maturity', 'historical_vol', 'oi_change']
    target_col = 'implied_vol'
    
    print(f"训练集大小: {train_data.shape}")
    print(f"验证集大小: {val_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 4. 训练模型
    print("\n4. 训练模型")
    print("-" * 40)
    
    # 创建模型
    print("创建波动率预测模型...")
    
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
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close(fig)
    
    # 5. 评估模型
    print("\n5. 评估模型")
    print("-" * 40)
    
    # 准备测试数据
    X_test = test_data[feature_cols]
    y_test = test_data[target_col].values.reshape(-1, 1)
    
    # 绘制预测结果
    fig, ax = vol_predictor.plot_predictions(X_test, y_test, n_samples=50)
    plt.savefig(os.path.join(results_dir, 'prediction_vs_actual.png'))
    plt.close(fig)
    
    # 6. 预测未来波动率曲面
    print("\n6. 预测未来波动率曲面")
    print("-" * 40)
    
    # 准备波动率曲面数据
    vol_surface_data = DataProcessor.prepare_vol_surface_data(option_data)
    
    # 预测未来波动率曲面
    print("预测未来波动率曲面...")
    future_days = 5
    future_surfaces = vol_predictor.predict_vol_surface(vol_surface_data, future_days=future_days)
    
    # 可视化预测的波动率曲面
    for day in range(min(3, len(future_surfaces))):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        vol_predictor.plot_predicted_vol_surface(future_surfaces[day], day=day, ax=ax)
        plt.savefig(os.path.join(results_dir, f'predicted_vol_surface_day{day+1}.png'))
        plt.close(fig)
    
    # 7. 期权定价应用
    print("\n7. 期权定价应用")
    print("-" * 40)
    
    # 创建期权定价模型
    bs_model = BlackScholes(option_type='call')
    
    # 选择几个代表性的期权
    sample_options = [
        {"strike": 90, "maturity": 0.25, "name": "实值短期期权"},
        {"strike": 100, "maturity": 0.25, "name": "平值短期期权"},
        {"strike": 110, "maturity": 0.25, "name": "虚值短期期权"},
        {"strike": 100, "maturity": 1.0, "name": "平值长期期权"},
    ]
    
    # 预测未来5天的期权价格
    option_prices = []
    
    for option in sample_options:
        K = option["strike"]
        T = option["maturity"]
        option_name = option["name"]
        
        # 当前价格（使用当前波动率）
        current_vol = vol_predictor.get_implied_vol(K, T)
        current_price = bs_model.price(S0, K, T, current_vol)
        
        # 预测未来5天的价格
        future_prices = []
        future_vols = []
        
        for day in range(future_days):
            # 获取预测的波动率
            day_data = future_surfaces[day]
            vol_subset = day_data[(day_data['strike'] == K) & (day_data['maturity'] == T)]
            
            if len(vol_subset) > 0:
                predicted_vol = vol_subset[target_col].values[0]
                future_vols.append(predicted_vol)
                
                # 计算期权价格
                # 注意：实际应用中应该考虑时间衰减
                adjusted_T = max(0, T - (day + 1) / 252)
                future_price = bs_model.price(S0, K, adjusted_T, predicted_vol)
                future_prices.append(future_price)
        
        option_prices.append({
            "name": option_name,
            "strike": K,
            "maturity": T,
            "current_price": current_price,
            "current_vol": current_vol,
            "future_prices": future_prices,
            "future_vols": future_vols
        })
    
    # 输出期权价格预测结果
    print("\n期权价格预测结果:")
    for option in option_prices:
        print(f"\n{option['name']} (K={option['strike']}, T={option['maturity']:.2f}):")
        print(f"当前价格: {option['current_price']:.4f} (波动率: {option['current_vol']:.4f})")
        
        for day, (price, vol) in enumerate(zip(option['future_prices'], option['future_vols'])):
            print(f"第{day+1}天预测: {price:.4f} (波动率: {vol:.4f})")
    
    # 绘制期权价格变化
    plt.figure(figsize=(12, 6))
    
    for option in option_prices:
        prices = [option['current_price']] + option['future_prices']
        plt.plot(range(len(prices)), prices, 'o-', label=option['name'])
    
    plt.xlabel('天数')
    plt.ylabel('期权价格')
    plt.title('期权价格预测')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'predicted_option_prices.png'))
    plt.close()
    
    # 绘制波动率变化
    plt.figure(figsize=(12, 6))
    
    for option in option_prices:
        vols = [option['current_vol']] + option['future_vols']
        plt.plot(range(len(vols)), vols, 'o-', label=option['name'])
    
    plt.xlabel('天数')
    plt.ylabel('隐含波动率')
    plt.title('波动率预测')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'predicted_volatilities.png'))
    plt.close()
    
    print("\n完成！结果已保存到:", results_dir)
    print("=" * 60)


if __name__ == "__main__":
    run_vol_prediction_example() 