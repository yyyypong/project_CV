"""
局部波动率模型模块
实现了Dupire公式和局部波动率曲面构建方法
"""

import numpy as np
from scipy import interpolate
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pricing import BlackScholes
from config import RISK_FREE_RATE


class LocalVolatility:
    """局部波动率模型"""
    
    def __init__(self, risk_free_rate=None):
        """
        初始化局部波动率模型
        
        参数：
            risk_free_rate (float): 无风险利率，默认使用配置文件中的值
        """
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
        self.bs_model = BlackScholes(risk_free_rate=self.risk_free_rate)
        self.local_vol_surface = None
        self.strike_grid = None
        self.time_grid = None
    
    def calibrate(self, market_data, S0, method='spline'):
        """
        根据市场数据校准局部波动率曲面
        
        参数：
            market_data (DataFrame): 市场数据，包含列：'strike', 'maturity', 'price'
            S0 (float): 标的资产当前价格
            method (str): 插值方法，'spline'或'rbf'
            
        返回：
            LocalVolatility: 自身对象，支持链式调用
        """
        # 从市场数据中提取信息
        strikes = np.sort(market_data['strike'].unique())
        maturities = np.sort(market_data['maturity'].unique())
        
        # 创建隐含波动率网格
        implied_vol_grid = np.zeros((len(strikes), len(maturities)))
        
        # 计算隐含波动率
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                option_data = market_data[(market_data['strike'] == K) & 
                                          (market_data['maturity'] == T)]
                
                if len(option_data) > 0:
                    price = option_data['price'].values[0]
                    implied_vol = self.bs_model.implied_volatility(price, S0, K, T)
                    implied_vol_grid[i, j] = implied_vol
        
        # 计算局部波动率
        local_vol_grid = np.zeros_like(implied_vol_grid)
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma_imp = implied_vol_grid[i, j]
                
                # 如果隐含波动率无效，使用周围的平均值
                if np.isnan(sigma_imp) or sigma_imp <= 0:
                    neighbors = []
                    if i > 0 and not np.isnan(implied_vol_grid[i-1, j]):
                        neighbors.append(implied_vol_grid[i-1, j])
                    if i < len(strikes) - 1 and not np.isnan(implied_vol_grid[i+1, j]):
                        neighbors.append(implied_vol_grid[i+1, j])
                    if j > 0 and not np.isnan(implied_vol_grid[i, j-1]):
                        neighbors.append(implied_vol_grid[i, j-1])
                    if j < len(maturities) - 1 and not np.isnan(implied_vol_grid[i, j+1]):
                        neighbors.append(implied_vol_grid[i, j+1])
                    
                    if neighbors:
                        sigma_imp = np.mean(neighbors)
                    else:
                        # 如果没有有效的邻居，使用一个默认值
                        sigma_imp = 0.2
                
                # 使用改进的Dupire公式计算局部波动率
                local_vol = self._dupire_formula(S0, K, T, strikes, maturities, implied_vol_grid, i, j)
                local_vol_grid[i, j] = local_vol
        
        # 创建局部波动率曲面插值器
        if method == 'spline':
            self.local_vol_surface = interpolate.RectBivariateSpline(strikes, maturities, local_vol_grid)
        elif method == 'rbf':
            # 准备RBF插值的数据
            points = np.array([(K, T) for K in strikes for T in maturities])
            values = local_vol_grid.flatten()
            self.local_vol_surface = interpolate.Rbf(points[:, 0], points[:, 1], values, function='thin_plate')
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        # 保存网格供后续使用
        self.strike_grid = strikes
        self.time_grid = maturities
        
        return self
    
    def get_local_vol(self, S, K, T):
        """
        获取特定价格和到期时间的局部波动率
        
        参数：
            S (float): 当前资产价格
            K (float): 行权价格
            T (float): 到期时间（年）
            
        返回：
            float: 局部波动率
        """
        if self.local_vol_surface is None:
            raise RuntimeError("在获取局部波动率前必须先校准模型")
        
        # 边界检查，防止外推
        if isinstance(self.local_vol_surface, interpolate.RectBivariateSpline):
            K = max(min(K, self.strike_grid[-1]), self.strike_grid[0])
            T = max(min(T, self.time_grid[-1]), self.time_grid[0])
            return float(self.local_vol_surface(K, T))
        else:  # Rbf
            K = max(min(K, self.strike_grid[-1]), self.strike_grid[0])
            T = max(min(T, self.time_grid[-1]), self.time_grid[0])
            return float(self.local_vol_surface(K, T))
    
    def _dupire_formula(self, S0, K, T, strikes, maturities, implied_vol_grid, i, j):
        """
        使用Dupire公式计算局部波动率
        
        参数：
            S0 (float): 当前资产价格
            K (float): 行权价格
            T (float): 到期时间
            strikes (numpy.ndarray): 行权价格网格
            maturities (numpy.ndarray): 到期时间网格
            implied_vol_grid (numpy.ndarray): 隐含波动率网格
            i (int): 行权价格索引
            j (int): 到期时间索引
            
        返回：
            float: 局部波动率
        """
        # 获取中心点的隐含波动率
        sigma = implied_vol_grid[i, j]
        
        # 如果在边界上，返回隐含波动率作为局部波动率的近似
        if i == 0 or i == len(strikes) - 1 or j == 0 or j == len(maturities) - 1:
            return sigma
        
        # 计算偏导数
        dK = (strikes[i+1] - strikes[i-1]) / 2.0
        dT = (maturities[j+1] - maturities[j-1]) / 2.0
        
        # 二阶导数
        d2K = (implied_vol_grid[i+1, j] - 2*sigma + implied_vol_grid[i-1, j]) / (dK**2)
        
        # 一阶导数
        dK_sigma = (implied_vol_grid[i+1, j] - implied_vol_grid[i-1, j]) / (2*dK)
        dT_sigma = (implied_vol_grid[i, j+1] - implied_vol_grid[i, j-1]) / (2*dT)
        
        # 计算d1、d2
        moneyness = np.log(S0/K)
        d1 = (moneyness + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # Dupire公式
        numerator = 2 * (dT_sigma + (self.risk_free_rate - S0/K) * dK_sigma + 0.5 * d2K)
        denominator = (K**2 * (d1**2 - 1) * dK_sigma + K**2 * sigma * d2K)
        
        # 防止除以零
        if abs(denominator) < 1e-10:
            return sigma
        
        local_vol = sigma * np.sqrt(numerator / denominator)
        
        # 防止局部波动率过大或过小
        if np.isnan(local_vol) or local_vol <= 0 or local_vol > 3 * sigma:
            return sigma
        
        return local_vol
    
    def simulate_path(self, S0, T, steps, num_paths=1, seed=None):
        """
        使用局部波动率模型模拟资产价格路径
        
        参数：
            S0 (float): 初始资产价格
            T (float): 模拟时间长度（年）
            steps (int): 时间步数
            num_paths (int): 路径数量
            seed (int): 随机数种子
            
        返回：
            numpy.ndarray: 模拟的价格路径 (shape: (num_paths, steps+1))
        """
        if self.local_vol_surface is None:
            raise RuntimeError("在进行模拟前必须先校准模型")
        
        # 设置随机数种子（如果提供）
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化路径
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        
        # 初始化价格路径
        paths = np.zeros((num_paths, steps + 1))
        paths[:, 0] = S0
        
        # 使用Euler-Maruyama方法模拟
        for i in range(steps):
            t = i * dt
            
            for p in range(num_paths):
                S = paths[p, i]
                
                # 获取当前局部波动率
                local_vol = self.get_local_vol(S, S, t)
                
                # 生成随机增量
                dW = np.random.normal(0, 1)
                
                # 更新价格
                dS = self.risk_free_rate * S * dt + local_vol * S * sqrt_dt * dW
                paths[p, i+1] = S + dS
        
        return paths 