"""
配置文件，包含系统所需的常量和参数设置
"""

# 市场参数
RISK_FREE_RATE = 0.025  # 无风险利率
TRADING_DAYS_PER_YEAR = 252  # 每年交易日数量
TRADING_HOURS_PER_DAY = 6.5  # 每天交易小时数

# 期权定价参数
BLACK_SCHOLES = 'black_scholes'
HESTON = 'heston'
LOCAL_VOL = 'local_vol'
PRICING_MODEL = LOCAL_VOL  # 默认使用局部波动率模型

# 数值计算参数
DELTA_TIME = 1.0 / (TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY)  # 时间步长
MONTE_CARLO_PATHS = 10000  # 蒙特卡洛模拟路径数
FD_TIME_STEPS = 100  # 有限差分时间步数
FD_ASSET_STEPS = 200  # 有限差分资产价格步数

# 波动率曲面参数
STRIKE_POINTS = 21  # 行权价格点数
MATURITY_POINTS = 10  # 到期时间点数
VOL_SURFACE_METHOD = 'cubic'  # 插值方法：cubic, linear, rbf

# 对冲参数
DEFAULT_HEDGE_FREQ = 1  # 默认对冲频率（每天）
MIN_HEDGE_FREQ = 0.2  # 最小对冲频率（每5天一次）
MAX_HEDGE_FREQ = 10  # 最大对冲频率（每天10次）
GAMMA_THRESHOLD = 0.1  # Gamma对冲阈值
VEGA_THRESHOLD = 0.05  # Vega对冲阈值

# 机器学习参数
NN_HIDDEN_LAYERS = [64, 32, 16]  # 神经网络隐藏层
TRAINING_EPOCHS = 1000  # 训练轮数
BATCH_SIZE = 64  # 批次大小
LEARNING_RATE = 0.001  # 学习率
TEST_SIZE = 0.2  # 测试集比例
VALIDATION_SIZE = 0.1  # 验证集比例
FEATURE_COLUMNS = ['moneyness', 'time_to_maturity', 'historical_vol', 'oi_change']  # 特征列

# 数据参数
DATA_DIR = '../data'  # 数据目录
RESULTS_DIR = '../results'  # 结果目录
LOG_FILE = 'hedge_system.log'  # 日志文件 