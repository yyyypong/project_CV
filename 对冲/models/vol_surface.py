"""
波动率曲面建模模块
实现了隐含波动率曲面构建、校准和可视化
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pricing import BlackScholes
from config import VOL_SURFACE_METHOD, STRIKE_POINTS, MATURITY_POINTS


class VolatilitySurface:
    """波动率曲面模型"""
    
    def __init__(self, method=None):
        """
        初始化波动率曲面模型
        
        参数：
            method (str): 插值方法，'cubic', 'linear', 'rbf'
        """
        self.method = method or VOL_SURFACE_METHOD
        self.surface = None
        self.bs_model = BlackScholes()
        self.strike_grid = None
        self.time_grid = None
        self.data = None
    
    def calibrate(self, market_data, S0):
        """
        根据市场数据校准波动率曲面
        
        参数：
            market_data (DataFrame): 市场数据，包含列：'strike', 'maturity', 'price', 'option_type'
            S0 (float): 标的资产当前价格
            
        返回：
            VolatilitySurface: 自身对象，支持链式调用
        """
        # 保存数据供后续使用
        self.data = market_data.copy()
        self.S0 = S0
        
        # 计算隐含波动率
        self.data['implied_vol'] = np.nan
        
        for idx, row in self.data.iterrows():
            price = row['price']
            K = row['strike']
            T = row['maturity']
            option_type = row['option_type']
            
            # 设置期权类型
            self.bs_model.option_type = option_type
            
            # 计算隐含波动率
            try:
                implied_vol = self.bs_model.implied_volatility(price, S0, K, T)
                self.data.loc[idx, 'implied_vol'] = implied_vol
            except:
                # 如果计算失败，记录为NaN
                pass
        
        # 添加归一化行权价（moneyness）
        self.data['moneyness'] = self.data['strike'] / S0
        
        # 创建网格
        strikes = np.sort(self.data['strike'].unique())
        maturities = np.sort(self.data['maturity'].unique())
        
        # 保存原始网格
        self.strike_grid = strikes
        self.time_grid = maturities
        
        # 创建更密集的网格用于插值
        self.dense_strike_grid = np.linspace(min(strikes), max(strikes), STRIKE_POINTS)
        self.dense_time_grid = np.linspace(min(maturities), max(maturities), MATURITY_POINTS)
        
        # 创建插值器
        if self.method in ['cubic', 'linear']:
            # 准备插值的网格数据
            vol_grid = np.zeros((len(maturities), len(strikes)))
            
            for i, T in enumerate(maturities):
                for j, K in enumerate(strikes):
                    subset = self.data[(self.data['maturity'] == T) & 
                                       (self.data['strike'] == K)]
                    if len(subset) > 0 and not np.isnan(subset['implied_vol'].values[0]):
                        vol_grid[i, j] = subset['implied_vol'].values[0]
                    else:
                        # 如果没有数据，使用均值
                        vol_grid[i, j] = self.data['implied_vol'].mean()
            
            # 创建插值器
            self.surface = interpolate.RectBivariateSpline(
                maturities, strikes, vol_grid, kx=min(3, len(maturities)-1), ky=min(3, len(strikes)-1)
            )
        
        elif self.method == 'rbf':
            # 准备RBF插值的数据
            valid_data = self.data.dropna(subset=['implied_vol'])
            points = valid_data[['maturity', 'strike']].values
            values = valid_data['implied_vol'].values
            
            # 创建RBF插值器
            self.surface = interpolate.Rbf(
                points[:, 0], points[:, 1], values, function='thin_plate'
            )
        
        else:
            raise ValueError(f"不支持的插值方法: {self.method}")
        
        return self
    
    def get_implied_vol(self, K, T):
        """
        获取特定行权价和到期时间的隐含波动率
        
        参数：
            K (float): 行权价格
            T (float): 到期时间（年）
            
        返回：
            float: 隐含波动率
        """
        if self.surface is None:
            raise RuntimeError("在获取隐含波动率前必须先校准模型")
        
        # 边界检查，防止外推
        K = max(min(K, self.strike_grid[-1]), self.strike_grid[0])
        T = max(min(T, self.time_grid[-1]), self.time_grid[0])
        
        if self.method in ['cubic', 'linear']:
            return float(self.surface(T, K))
        else:  # rbf
            return float(self.surface(T, K))
    
    def get_vol_slice(self, fixed_var, var_type='T'):
        """
        获取波动率曲面的一个切片
        
        参数：
            fixed_var (float): 固定的变量值（T或K）
            var_type (str): 变量类型，'T'表示固定到期时间，'K'表示固定行权价
            
        返回：
            tuple: (x, y) - x是变化的变量，y是对应的波动率值
        """
        if self.surface is None:
            raise RuntimeError("在获取波动率切片前必须先校准模型")
        
        if var_type == 'T':
            # 固定到期时间，变化行权价
            T = fixed_var
            strikes = self.dense_strike_grid
            vols = np.array([self.get_implied_vol(K, T) for K in strikes])
            return strikes, vols
        
        elif var_type == 'K':
            # 固定行权价，变化到期时间
            K = fixed_var
            maturities = self.dense_time_grid
            vols = np.array([self.get_implied_vol(K, T) for T in maturities])
            return maturities, vols
        
        else:
            raise ValueError(f"不支持的变量类型: {var_type}")
    
    def plot_surface(self, ax=None, alpha=0.7, cmap='viridis'):
        """
        绘制波动率曲面三维图
        
        参数：
            ax (matplotlib.axes.Axes): 用于绘图的坐标轴，如果为None则创建新的
            alpha (float): 透明度
            cmap (str): 色图
            
        返回：
            matplotlib.axes.Axes: 绘图使用的坐标轴
        """
        if self.surface is None:
            raise RuntimeError("在绘制波动率曲面前必须先校准模型")
        
        # 创建网格
        K, T = np.meshgrid(self.dense_strike_grid, self.dense_time_grid)
        
        # 计算每个网格点的波动率
        Z = np.zeros_like(K)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                Z[i, j] = self.get_implied_vol(K[i, j], T[i, j])
        
        # 创建图像
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # 绘制曲面
        surf = ax.plot_surface(K, T, Z, cmap=cmap, alpha=alpha, linewidth=0, antialiased=True)
        
        # 添加标签
        ax.set_xlabel('行权价 (K)')
        ax.set_ylabel('到期时间 (T)')
        ax.set_zlabel('隐含波动率 (\sigma)')
        ax.set_title('隐含波动率曲面')
        
        # 添加颜色条
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return ax
    
    def plot_smile(self, maturities=None, ax=None):
        """
        绘制波动率微笑曲线
        
        参数：
            maturities (list): 要绘制的到期时间列表，如果为None则使用数据中的所有值
            ax (matplotlib.axes.Axes): 用于绘图的坐标轴，如果为None则创建新的
            
        返回：
            matplotlib.axes.Axes: 绘图使用的坐标轴
        """
        if self.surface is None:
            raise RuntimeError("在绘制波动率微笑前必须先校准模型")
        
        if maturities is None:
            maturities = [min(self.time_grid), np.median(self.time_grid), max(self.time_grid)]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for T in maturities:
            strikes, vols = self.get_vol_slice(T, var_type='T')
            moneyness = strikes / self.S0
            ax.plot(moneyness, vols, '-', label=f'T = {T:.2f}')
        
        ax.set_xlabel('价格比率 (K/S0)')
        ax.set_ylabel('隐含波动率 (\sigma)')
        ax.set_title('波动率微笑曲线')
        ax.legend()
        ax.grid(True)
        
        return ax
    
    def plot_term_structure(self, moneyness_levels=None, ax=None):
        """
        绘制波动率期限结构
        
        参数：
            moneyness_levels (list): 要绘制的价格比率列表，如果为None则使用一些默认值
            ax (matplotlib.axes.Axes): 用于绘图的坐标轴，如果为None则创建新的
            
        返回：
            matplotlib.axes.Axes: 绘图使用的坐标轴
        """
        if self.surface is None:
            raise RuntimeError("在绘制波动率期限结构前必须先校准模型")
        
        if moneyness_levels is None:
            moneyness_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        for m in moneyness_levels:
            K = m * self.S0
            times, vols = self.get_vol_slice(K, var_type='K')
            ax.plot(times, vols, '-', label=f'K/S0 = {m:.2f}')
        
        ax.set_xlabel('到期时间 (T)')
        ax.set_ylabel('隐含波动率 (\sigma)')
        ax.set_title('波动率期限结构')
        ax.legend()
        ax.grid(True)
        
        return ax
    
    def export_surface_data(self):
        """
        导出波动率曲面数据
        
        返回：
            pandas.DataFrame: 包含网格点和对应波动率的数据框
        """
        if self.surface is None:
            raise RuntimeError("在导出波动率曲面数据前必须先校准模型")
        
        # 创建网格
        strike_grid = self.dense_strike_grid
        time_grid = self.dense_time_grid
        
        # 创建结果数据框
        result = []
        
        for T in time_grid:
            for K in strike_grid:
                vol = self.get_implied_vol(K, T)
                result.append({
                    'maturity': T,
                    'strike': K,
                    'moneyness': K / self.S0,
                    'implied_vol': vol
                })
        
        return pd.DataFrame(result) 