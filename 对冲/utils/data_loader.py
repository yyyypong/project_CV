"""
数据加载和处理工具
包括市场数据获取、清洗和预处理功能
"""

import pandas as pd
import numpy as np
import os
import sys
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, RISK_FREE_RATE
from models.pricing import BlackScholes


class DataLoader:
    """数据加载类"""
    
    def __init__(self, data_dir=None):
        """
        初始化数据加载器
        
        参数：
            data_dir (str): 数据目录路径，默认使用配置文件中的值
        """
        self.data_dir = data_dir or DATA_DIR
    
    def load_option_data(self, file_path=None, ticker='SPY'):
        """
        加载期权市场数据
        
        参数：
            file_path (str): 文件路径，如果为None则使用默认路径
            ticker (str): 股票代码，用于构建默认文件路径
            
        返回：
            pandas.DataFrame: 期权数据
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, f"{ticker}_options.csv")
        
        try:
            data = pd.read_csv(file_path)
            print(f"成功加载期权数据: {file_path}")
            return data
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            
            # 创建模拟数据
            return self.generate_mock_option_data(ticker)
    
    def load_stock_data(self, file_path=None, ticker='SPY'):
        """
        加载股票市场数据
        
        参数：
            file_path (str): 文件路径，如果为None则使用默认路径
            ticker (str): 股票代码，用于构建默认文件路径
            
        返回：
            pandas.DataFrame: 股票数据
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, f"{ticker}_stock.csv")
        
        try:
            data = pd.read_csv(file_path)
            print(f"成功加载股票数据: {file_path}")
            return data
        except FileNotFoundError:
            print(f"文件不存在: {file_path}")
            
            # 创建模拟数据
            return self.generate_mock_stock_data(ticker)
    
    def generate_mock_option_data(self, ticker='SPY'):
        """
        生成模拟期权数据
        
        参数：
            ticker (str): 股票代码
            
        返回：
            pandas.DataFrame: 模拟期权数据
        """
        print("生成模拟期权数据...")
        
        # 设置参数
        S0 = 100.0  # 当前股价
        vol_levels = {
            30: 0.2,    # 30天期限的波动率
            60: 0.22,   # 60天期限的波动率
            90: 0.24,   # 90天期限的波动率
            180: 0.26,  # 180天期限的波动率
            360: 0.28   # 360天期限的波动率
        }
        
        # 生成日期范围
        today = dt.date.today()
        dates = [today - dt.timedelta(days=i) for i in range(30)]
        
        # 生成行权价格范围，从70%到130%的股价
        strikes = np.linspace(0.7 * S0, 1.3 * S0, 13)
        
        # 生成到期日期范围
        maturities = list(vol_levels.keys())
        
        # 创建定价模型
        bs_call = BlackScholes(option_type='call')
        bs_put = BlackScholes(option_type='put')
        
        # 生成数据
        records = []
        
        for date in dates:
            # 随机生成当天的基础波动率变动
            base_vol_change = np.random.normal(0, 0.01)
            
            for maturity_days in maturities:
                maturity_date = date + dt.timedelta(days=maturity_days)
                # 到期时间（年）
                T = maturity_days / 365.0
                
                # 当天该期限的波动率
                base_vol = vol_levels[maturity_days]
                daily_vol = max(0.1, base_vol + base_vol_change)
                
                for K in strikes:
                    # 添加微笑效应：深度实值和虚值期权波动率较高
                    moneyness = K / S0
                    smile_factor = 0.05 * (moneyness - 1) ** 2
                    
                    # 最终波动率
                    sigma = daily_vol + smile_factor
                    
                    # 计算期权价格
                    call_price = bs_call.price(S0, K, T, sigma)
                    put_price = bs_put.price(S0, K, T, sigma)
                    
                    # 添加随机噪声
                    call_price *= (1 + np.random.normal(0, 0.01))
                    put_price *= (1 + np.random.normal(0, 0.01))
                    
                    # 记录看涨期权数据
                    records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'strike': K,
                        'expiration': maturity_date.strftime('%Y-%m-%d'),
                        'option_type': 'call',
                        'price': call_price,
                        'implied_vol': sigma,
                        'open_interest': int(np.random.uniform(100, 1000)),
                        'volume': int(np.random.uniform(10, 500))
                    })
                    
                    # 记录看跌期权数据
                    records.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'strike': K,
                        'expiration': maturity_date.strftime('%Y-%m-%d'),
                        'option_type': 'put',
                        'price': put_price,
                        'implied_vol': sigma,
                        'open_interest': int(np.random.uniform(100, 1000)),
                        'volume': int(np.random.uniform(10, 500))
                    })
        
        # 创建DataFrame
        df = pd.DataFrame(records)
        
        # 保存数据
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, f"{ticker}_options.csv"), index=False)
        
        return df
    
    def generate_mock_stock_data(self, ticker='SPY'):
        """
        生成模拟股票数据
        
        参数：
            ticker (str): 股票代码
            
        返回：
            pandas.DataFrame: 模拟股票数据
        """
        print("生成模拟股票数据...")
        
        # 设置参数
        S0 = 100.0  # 起始价格
        vol = 0.2  # 年化波动率
        
        # 生成日期范围
        today = dt.date.today()
        dates = [today - dt.timedelta(days=i) for i in range(252)]  # 一年的交易日
        dates = sorted(dates)  # 按日期升序排序
        
        # 生成股价路径
        np.random.seed(42)  # 设置随机种子以便重现
        
        # 日收益率的标准差
        daily_vol = vol / np.sqrt(252)
        
        # 生成日收益率
        daily_returns = np.random.normal(RISK_FREE_RATE / 252, daily_vol, len(dates) - 1)
        
        # 计算股价
        prices = [S0]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # 计算每日交易量
        volumes = [int(np.random.uniform(1000000, 5000000)) for _ in range(len(dates))]
        
        # 计算每日最高和最低价格
        highs = [price * (1 + np.random.uniform(0, 0.02)) for price in prices]
        lows = [price * (1 - np.random.uniform(0, 0.02)) for price in prices]
        
        # 创建数据记录
        records = []
        for i, date in enumerate(dates):
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'open': prices[i],
                'high': highs[i],
                'low': lows[i],
                'close': prices[i],
                'volume': volumes[i]
            })
        
        # 创建DataFrame
        df = pd.DataFrame(records)
        
        # 保存数据
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(os.path.join(self.data_dir, f"{ticker}_stock.csv"), index=False)
        
        return df


class DataProcessor:
    """数据处理类"""
    
    @staticmethod
    def process_option_data(data):
        """
        处理期权数据
        
        参数：
            data (DataFrame): 原始期权数据
            
        返回：
            DataFrame: 处理后的数据
        """
        df = data.copy()
        
        # 转换日期列
        date_columns = ['date', 'expiration']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # 计算到期时间（年）
        if 'date' in df.columns and 'expiration' in df.columns:
            df['time_to_maturity'] = (df['expiration'] - df['date']).dt.days / 365.0
            
            # 去除已过期的期权
            df = df[df['time_to_maturity'] > 0]
        
        # 标准化期权类型
        if 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.lower()
        
        # 添加是否虚值/实值标志
        if 'strike' in df.columns and 'underlying_price' in df.columns:
            df['moneyness'] = df['strike'] / df['underlying_price']
            
            # ATM: 0.95 < moneyness < 1.05
            # ITM for call: moneyness < 0.95
            # OTM for call: moneyness > 1.05
            # OTM for put: moneyness < 0.95
            # ITM for put: moneyness > 1.05
            
            df['money_status'] = 'ATM'  # At-the-money
            
            # 看涨期权
            call_idx = df['option_type'] == 'call'
            df.loc[call_idx & (df['moneyness'] < 0.95), 'money_status'] = 'ITM'  # In-the-money
            df.loc[call_idx & (df['moneyness'] > 1.05), 'money_status'] = 'OTM'  # Out-of-the-money
            
            # 看跌期权
            put_idx = df['option_type'] == 'put'
            df.loc[put_idx & (df['moneyness'] < 0.95), 'money_status'] = 'OTM'
            df.loc[put_idx & (df['moneyness'] > 1.05), 'money_status'] = 'ITM'
        
        return df
    
    @staticmethod
    def calculate_historical_volatility(stock_data, window=30):
        """
        计算历史波动率
        
        参数：
            stock_data (DataFrame): 股票数据
            window (int): 计算窗口大小（天数）
            
        返回：
            DataFrame: 包含历史波动率的数据
        """
        # 确保数据按日期排序
        df = stock_data.sort_values('date')
        
        # 计算日收益率
        df['returns'] = df['close'].pct_change()
        
        # 计算滚动标准差
        df['historical_vol'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def prepare_vol_surface_data(option_data, date=None):
        """
        准备构建波动率曲面的数据
        
        参数：
            option_data (DataFrame): 期权数据
            date (str or datetime): 指定日期，默认使用最新日期
            
        返回：
            DataFrame: 波动率曲面数据
        """
        df = option_data.copy()
        
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 筛选指定日期的数据
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            df = df[df['date'] == date]
        else:
            # 使用最新日期
            latest_date = df['date'].max()
            df = df[df['date'] == latest_date]
        
        # 确保必要的列存在
        required_columns = ['strike', 'time_to_maturity', 'implied_vol', 'option_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少以下列: {missing_columns}")
        
        # 计算平均隐含波动率（合并看涨和看跌期权）
        vol_data = df.groupby(['strike', 'time_to_maturity']).agg({
            'implied_vol': 'mean',
            'price': 'mean',
            'option_type': 'first'  # 保留任意一个期权类型
        }).reset_index()
        
        return vol_data
    
    @staticmethod
    def create_ml_features(option_data, stock_data, lookback_days=5):
        """
        创建用于机器学习的特征
        
        参数：
            option_data (DataFrame): 期权数据
            stock_data (DataFrame): 股票数据
            lookback_days (int): 回顾天数，用于计算变化率特征
            
        返回：
            DataFrame: 包含特征的数据
        """
        # 确保日期列是datetime类型
        option_df = option_data.copy()
        stock_df = stock_data.copy()
        
        option_df['date'] = pd.to_datetime(option_df['date'])
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        # 合并股票数据
        df = pd.merge(option_df, stock_df[['date', 'historical_vol', 'returns']], 
                     on='date', how='left')
        
        # 计算隐含波动率变化率
        df = df.sort_values(['date', 'strike', 'time_to_maturity'])
        
        # 按期权分组计算隐含波动率变化
        df['implied_vol_change'] = df.groupby(['strike', 'option_type'])['implied_vol'].pct_change(lookback_days)
        
        # 计算未平仓合约变化率
        if 'open_interest' in df.columns:
            df['oi_change'] = df.groupby(['strike', 'option_type'])['open_interest'].pct_change(lookback_days)
        
        # 计算交易量变化率
        if 'volume' in df.columns:
            df['volume_change'] = df.groupby(['strike', 'option_type'])['volume'].pct_change(lookback_days)
        
        # 计算价格/波动率比率（P/V比）
        df['price_vol_ratio'] = df['price'] / df['implied_vol']
        
        # 计算实现波动率与隐含波动率之差
        df['vol_premium'] = df['implied_vol'] - df['historical_vol']
        
        # 删除NaN值
        df = df.dropna(subset=['implied_vol_change'])
        
        return df
    
    @staticmethod
    def split_option_data(df, test_size=0.2, validation_size=0.1, random_state=42):
        """
        将期权数据分割为训练集、验证集和测试集
        
        参数：
            df (DataFrame): 期权数据
            test_size (float): 测试集比例
            validation_size (float): 验证集比例
            random_state (int): 随机种子
            
        返回：
            tuple: (train_df, val_df, test_df)
        """
        # 确保数据按日期排序
        df = df.sort_values('date')
        
        # 获取唯一日期
        dates = df['date'].unique()
        
        # 计算分割点
        n_dates = len(dates)
        test_start = int(n_dates * (1 - test_size))
        val_start = int(n_dates * (1 - test_size - validation_size))
        
        # 分割数据
        train_dates = dates[:val_start]
        val_dates = dates[val_start:test_start]
        test_dates = dates[test_start:]
        
        # 创建数据集
        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        print(f"训练集: {len(train_df)} 行, {len(train_dates)} 天")
        print(f"验证集: {len(val_df)} 行, {len(val_dates)} 天")
        print(f"测试集: {len(test_df)} 行, {len(test_dates)} 天")
        
        return train_df, val_df, test_df 