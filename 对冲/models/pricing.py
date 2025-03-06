"""
期权定价模型模块
包含Black-Scholes公式、二叉树、蒙特卡洛方法等多种期权定价算法
"""

import numpy as np
from scipy.stats import norm
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, DELTA_TIME, MONTE_CARLO_PATHS, FD_TIME_STEPS, FD_ASSET_STEPS


class OptionPricing:
    """期权定价基类"""
    
    def __init__(self, option_type='call', risk_free_rate=None):
        """
        初始化期权定价模型
        
        参数：
            option_type (str): 期权类型，'call'或'put'
            risk_free_rate (float): 无风险利率，默认使用配置文件中的值
        """
        self.option_type = option_type.lower()
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
    
    def price(self, *args, **kwargs):
        """
        期权定价方法（由子类实现）
        
        返回：
            float: 期权价格
        """
        raise NotImplementedError("子类必须实现此方法")


class BlackScholes(OptionPricing):
    """Black-Scholes期权定价模型"""
    
    def price(self, S, K, T, sigma, dividend=0.0):
        """
        使用Black-Scholes公式计算期权价格
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            dividend (float): 分红率
            
        返回：
            float: 期权价格
        """
        if T <= 0:
            # 已到期期权的价值
            if self.option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
                
        # 调整无风险利率（考虑分红）
        r = self.risk_free_rate - dividend
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if self.option_type == 'call':
            price = S * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * np.exp(-dividend * T) * norm.cdf(-d1)
            
        return price
    
    def implied_volatility(self, price, S, K, T, initial_vol=0.2, max_iterations=100, precision=1e-8):
        """
        计算隐含波动率
        
        参数：
            price (float): 市场期权价格
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            initial_vol (float): 初始波动率猜测值
            max_iterations (int): 最大迭代次数
            precision (float): 收敛精度
            
        返回：
            float: 隐含波动率
        """
        if T <= 0:
            return np.nan
            
        vol = initial_vol
        
        # 牛顿-拉弗森法求解隐含波动率
        for _ in range(max_iterations):
            # 计算当前波动率下的期权价格
            price_diff = self.price(S, K, T, vol) - price
            
            if abs(price_diff) < precision:
                return vol
                
            # 计算波动率变化对期权价格的影响（vega）
            vega = self._vega(S, K, T, vol)
            
            # 防止除零
            if abs(vega) < 1e-10:
                break
                
            # 更新波动率
            vol = vol - price_diff / vega
            
            # 防止波动率为负或过大
            if vol <= 0 or vol > 5:
                vol = initial_vol
                break
                
        return vol
    
    def _vega(self, S, K, T, sigma, dividend=0.0):
        """
        计算vega（期权价格对波动率的敏感性）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            dividend (float): 分红率
            
        返回：
            float: vega值
        """
        if T <= 0:
            return 0
            
        r = self.risk_free_rate - dividend
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        vega = S * np.exp(-dividend * T) * norm.pdf(d1) * np.sqrt(T)
        return vega


class MonteCarlo(OptionPricing):
    """蒙特卡洛模拟定价模型"""
    
    def price(self, S, K, T, sigma, dividend=0.0, paths=None):
        """
        使用蒙特卡洛模拟计算期权价格
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            dividend (float): 分红率
            paths (int): 模拟路径数量
            
        返回：
            float: 期权价格
        """
        if T <= 0:
            # 已到期期权的价值
            if self.option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        paths = paths or MONTE_CARLO_PATHS
        dt = DELTA_TIME
        steps = int(T / dt)
        
        # 调整无风险利率（考虑分红）
        r = self.risk_free_rate - dividend
        
        # 生成随机路径
        Z = np.random.normal(0, 1, (paths, steps))
        delta_S = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # 计算资产价格路径
        S_paths = np.zeros((paths, steps + 1))
        S_paths[:, 0] = S
        
        for i in range(steps):
            S_paths[:, i+1] = S_paths[:, i] * delta_S[:, i]
        
        # 计算期权价值
        if self.option_type == 'call':
            payoffs = np.maximum(0, S_paths[:, -1] - K)
        else:  # put
            payoffs = np.maximum(0, K - S_paths[:, -1])
        
        # 折现
        option_price = np.exp(-self.risk_free_rate * T) * np.mean(payoffs)
        
        return option_price


class FiniteDifference(OptionPricing):
    """有限差分法定价模型"""
    
    def price(self, S, K, T, sigma, dividend=0.0, time_steps=None, asset_steps=None):
        """
        使用有限差分法（隐式差分）计算期权价格
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            dividend (float): 分红率
            time_steps (int): 时间步数
            asset_steps (int): 资产价格步数
            
        返回：
            float: 期权价格
        """
        if T <= 0:
            # 已到期期权的价值
            if self.option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # 参数设置
        time_steps = time_steps or FD_TIME_STEPS
        asset_steps = asset_steps or FD_ASSET_STEPS
        
        # 调整无风险利率（考虑分红）
        r = self.risk_free_rate - dividend
        
        # 设置网格
        dt = T / time_steps
        
        # 价格上下限
        S_max = S * 3  # 上限设为当前价格的3倍
        S_min = S / 3  # 下限设为当前价格的1/3
        
        dS = (S_max - S_min) / asset_steps
        
        # 创建网格
        grid = np.zeros((asset_steps + 1, time_steps + 1))
        S_values = np.linspace(S_min, S_max, asset_steps + 1)
        
        # 设置到期边界条件
        if self.option_type == 'call':
            grid[:, -1] = np.maximum(0, S_values - K)
        else:  # put
            grid[:, -1] = np.maximum(0, K - S_values)
        
        # 设置空间边界条件
        for j in range(time_steps, -1, -1):
            if self.option_type == 'call':
                grid[0, j] = 0  # 当资产价格接近0时，看涨期权价值为0
                grid[-1, j] = S_max - K * np.exp(-r * dt * (time_steps - j))  # 当资产价格很高时的渐近线
            else:  # put
                grid[0, j] = K * np.exp(-r * dt * (time_steps - j))  # 当资产价格接近0时，看跌期权价值接近贴现的行权价
                grid[-1, j] = 0  # 当资产价格很高时，看跌期权价值为0
        
        # 三对角矩阵求解器
        A = np.zeros((asset_steps - 1, asset_steps - 1))
        b = np.zeros(asset_steps - 1)
        
        # 使用隐式差分法计算期权价格
        for j in range(time_steps - 1, -1, -1):
            for i in range(asset_steps - 1):
                S_i = S_min + (i + 1) * dS
                
                # 构建差分方程系数
                alpha = 0.5 * dt * (sigma**2 * S_i**2 / dS**2 - r * S_i / dS)
                beta = 1 - dt * (sigma**2 * S_i**2 / dS**2 + r)
                gamma = 0.5 * dt * (sigma**2 * S_i**2 / dS**2 + r * S_i / dS)
                
                # 设置三对角矩阵系数
                if i > 0:
                    A[i, i-1] = alpha
                A[i, i] = beta
                if i < asset_steps - 2:
                    A[i, i+1] = gamma
                
                # 右侧向量
                b[i] = grid[i+1, j+1]
                
            # 处理边界条件的影响
            b[0] -= alpha * grid[0, j]
            b[-1] -= gamma * grid[-1, j]
            
            # 解线性方程组
            x = np.linalg.solve(A, b)
            
            # 更新网格
            grid[1:-1, j] = x
        
        # 通过插值获取当前股价对应的期权价格
        option_price = np.interp(S, S_values, grid[:, 0])
        
        return option_price 