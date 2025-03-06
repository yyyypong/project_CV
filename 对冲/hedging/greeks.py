"""
希腊字母计算模块
实现了期权的各种希腊字母（Delta、Gamma、Vega、Theta、Rho）的计算
"""

import numpy as np
from scipy.stats import norm
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pricing import BlackScholes, FiniteDifference
from models.local_vol import LocalVolatility
from config import RISK_FREE_RATE


class GreeksCalculator:
    """期权希腊字母计算器"""
    
    def __init__(self, pricing_model=None, risk_free_rate=None):
        """
        初始化希腊字母计算器
        
        参数：
            pricing_model (OptionPricing): 定价模型，默认使用Black-Scholes
            risk_free_rate (float): 无风险利率，默认使用配置文件中的值
        """
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
        self.pricing_model = pricing_model if pricing_model is not None else BlackScholes(risk_free_rate=self.risk_free_rate)
    
    def delta(self, S, K, T, sigma, option_type='call', method='analytical', h=0.01, dividend=0.0):
        """
        计算Delta（期权价格对标的资产价格的一阶导数）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            h (float): 有限差分步长
            dividend (float): 分红率
            
        返回：
            float: Delta值
        """
        if method == 'analytical' and isinstance(self.pricing_model, BlackScholes):
            # 使用Black-Scholes公式的解析解
            if T <= 0:
                if option_type == 'call':
                    return 1.0 if S > K else 0.0
                else:  # put
                    return -1.0 if S < K else 0.0
            
            # 调整无风险利率（考虑分红）
            r = self.risk_free_rate - dividend
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if option_type == 'call':
                delta = np.exp(-dividend * T) * norm.cdf(d1)
            else:  # put
                delta = np.exp(-dividend * T) * (norm.cdf(d1) - 1)
                
            return delta
        else:
            # 使用有限差分法
            self.pricing_model.option_type = option_type
            
            # 中心差分
            price_up = self.pricing_model.price(S + h, K, T, sigma, dividend)
            price_down = self.pricing_model.price(S - h, K, T, sigma, dividend)
            
            delta = (price_up - price_down) / (2 * h)
            return delta
    
    def gamma(self, S, K, T, sigma, option_type='call', method='analytical', h=0.01, dividend=0.0):
        """
        计算Gamma（Delta对标的资产价格的导数）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            h (float): 有限差分步长
            dividend (float): 分红率
            
        返回：
            float: Gamma值
        """
        if method == 'analytical' and isinstance(self.pricing_model, BlackScholes):
            # 使用Black-Scholes公式的解析解
            if T <= 0:
                return 0.0
            
            # 调整无风险利率（考虑分红）
            r = self.risk_free_rate - dividend
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            gamma = np.exp(-dividend * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            return gamma
        else:
            # 使用有限差分法
            self.pricing_model.option_type = option_type
            
            # 二阶中心差分
            price_up = self.pricing_model.price(S + h, K, T, sigma, dividend)
            price_center = self.pricing_model.price(S, K, T, sigma, dividend)
            price_down = self.pricing_model.price(S - h, K, T, sigma, dividend)
            
            gamma = (price_up - 2 * price_center + price_down) / (h**2)
            return gamma
    
    def vega(self, S, K, T, sigma, option_type='call', method='analytical', h=0.001, dividend=0.0):
        """
        计算Vega（期权价格对波动率的导数）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            h (float): 有限差分步长
            dividend (float): 分红率
            
        返回：
            float: Vega值（通常表示为波动率变化1%对应的期权价格变化）
        """
        if method == 'analytical' and isinstance(self.pricing_model, BlackScholes):
            # 使用Black-Scholes公式的解析解
            if T <= 0:
                return 0.0
            
            # 调整无风险利率（考虑分红）
            r = self.risk_free_rate - dividend
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            vega = S * np.exp(-dividend * T) * norm.pdf(d1) * np.sqrt(T)
            
            # 转换为每1%波动率变化的价格变化
            vega = vega / 100
            
            return vega
        else:
            # 使用有限差分法
            self.pricing_model.option_type = option_type
            
            # 中心差分
            price_up = self.pricing_model.price(S, K, T, sigma + h, dividend)
            price_down = self.pricing_model.price(S, K, T, sigma - h, dividend)
            
            vega = (price_up - price_down) / (2 * h)
            
            # 转换为每1%波动率变化的价格变化
            vega = vega / 100
            
            return vega
    
    def theta(self, S, K, T, sigma, option_type='call', method='analytical', h=0.001, dividend=0.0):
        """
        计算Theta（期权价格对时间的导数）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            h (float): 有限差分步长
            dividend (float): 分红率
            
        返回：
            float: Theta值（每天的价格变化）
        """
        if method == 'analytical' and isinstance(self.pricing_model, BlackScholes):
            # 使用Black-Scholes公式的解析解
            if T <= 0:
                return 0.0
            
            # 调整无风险利率（考虑分红）
            r = self.risk_free_rate - dividend
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                theta = -S * np.exp(-dividend * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - \
                        r * K * np.exp(-r * T) * norm.cdf(d2) + \
                        dividend * S * np.exp(-dividend * T) * norm.cdf(d1)
            else:  # put
                theta = -S * np.exp(-dividend * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + \
                        r * K * np.exp(-r * T) * norm.cdf(-d2) - \
                        dividend * S * np.exp(-dividend * T) * norm.cdf(-d1)
            
            # 转换为每天的价格变化（而不是每年）
            theta = theta / 365
            
            return theta
        else:
            # 使用有限差分法（向前差分）
            self.pricing_model.option_type = option_type
            
            price_now = self.pricing_model.price(S, K, T, sigma, dividend)
            price_later = self.pricing_model.price(S, K, T - h, sigma, dividend)
            
            theta = (price_later - price_now) / h
            
            # 转换为每天的价格变化
            theta = theta / 365
            
            return theta
    
    def rho(self, S, K, T, sigma, option_type='call', method='analytical', h=0.001, dividend=0.0):
        """
        计算Rho（期权价格对利率的导数）
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            h (float): 有限差分步长
            dividend (float): 分红率
            
        返回：
            float: Rho值（利率变化1%对应的期权价格变化）
        """
        if method == 'analytical' and isinstance(self.pricing_model, BlackScholes):
            # 使用Black-Scholes公式的解析解
            if T <= 0:
                return 0.0
            
            # 调整无风险利率（考虑分红）
            r = self.risk_free_rate - dividend
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            # 转换为每1%利率变化的价格变化
            rho = rho / 100
            
            return rho
        else:
            # 使用有限差分法
            original_rate = self.pricing_model.risk_free_rate
            
            # 设置利率上下变动
            self.pricing_model.risk_free_rate = original_rate + h
            self.pricing_model.option_type = option_type
            price_up = self.pricing_model.price(S, K, T, sigma, dividend)
            
            self.pricing_model.risk_free_rate = original_rate - h
            price_down = self.pricing_model.price(S, K, T, sigma, dividend)
            
            # 恢复原始利率
            self.pricing_model.risk_free_rate = original_rate
            
            rho = (price_up - price_down) / (2 * h)
            
            # 转换为每1%利率变化的价格变化
            rho = rho / 100
            
            return rho
    
    def all_greeks(self, S, K, T, sigma, option_type='call', method='analytical', dividend=0.0):
        """
        计算所有希腊字母
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            sigma (float): 波动率
            option_type (str): 期权类型，'call'或'put'
            method (str): 计算方法，'analytical'或'finite_diff'
            dividend (float): 分红率
            
        返回：
            dict: 包含所有希腊字母的字典
        """
        delta = self.delta(S, K, T, sigma, option_type, method, dividend=dividend)
        gamma = self.gamma(S, K, T, sigma, option_type, method, dividend=dividend)
        vega = self.vega(S, K, T, sigma, option_type, method, dividend=dividend)
        theta = self.theta(S, K, T, sigma, option_type, method, dividend=dividend)
        rho = self.rho(S, K, T, sigma, option_type, method, dividend=dividend)
        
        self.pricing_model.option_type = option_type
        price = self.pricing_model.price(S, K, T, sigma, dividend)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def local_vol_greeks(self, S, K, T, local_vol_model, option_type='call'):
        """
        使用局部波动率模型计算希腊字母
        
        参数：
            S (float): 标的资产当前价格
            K (float): 期权行权价格
            T (float): 期权到期时间（年）
            local_vol_model (LocalVolatility): 局部波动率模型
            option_type (str): 期权类型，'call'或'put'
            
        返回：
            dict: 包含所有希腊字母的字典
        """
        # 使用局部波动率计算隐含波动率
        sigma = local_vol_model.get_local_vol(S, K, T)
        
        # 使用有限差分定价模型
        fd_model = FiniteDifference(option_type=option_type, risk_free_rate=self.risk_free_rate)
        
        # 保存原始定价模型
        original_model = self.pricing_model
        
        # 设置有限差分模型
        self.pricing_model = fd_model
        
        # 计算希腊字母
        greeks = self.all_greeks(S, K, T, sigma, option_type, method='finite_diff')
        
        # 恢复原始定价模型
        self.pricing_model = original_model
        
        return greeks
    
    def portfolio_greeks(self, positions, S, local_vol_model=None):
        """
        计算投资组合的希腊字母
        
        参数：
            positions (list): 持仓列表，每个元素是一个字典，包含：
                              'type': 期权类型，'call'或'put'
                              'K': 行权价
                              'T': 到期时间
                              'sigma': 波动率（如果使用固定波动率模型）
                              'quantity': 持仓数量（正为多头，负为空头）
            S (float): 标的资产当前价格
            local_vol_model (LocalVolatility): 局部波动率模型，如果为None则使用固定波动率
            
        返回：
            dict: 包含投资组合希腊字母的字典
        """
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            option_type = position['type']
            K = position['K']
            T = position['T']
            quantity = position['quantity']
            
            if local_vol_model is not None:
                # 使用局部波动率模型
                greeks = self.local_vol_greeks(S, K, T, local_vol_model, option_type)
            else:
                # 使用固定波动率
                sigma = position['sigma']
                greeks = self.all_greeks(S, K, T, sigma, option_type)
            
            # 累加投资组合希腊字母
            for greek in portfolio_greeks:
                if greek in greeks:
                    portfolio_greeks[greek] += greeks[greek] * quantity
        
        return portfolio_greeks 