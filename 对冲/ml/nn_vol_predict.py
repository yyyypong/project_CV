"""
神经网络波动率预测模型
用于预测波动率曲面的动态变化和未来波动率
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NN_HIDDEN_LAYERS, TRAINING_EPOCHS, BATCH_SIZE, 
    LEARNING_RATE, TEST_SIZE, VALIDATION_SIZE, FEATURE_COLUMNS
)


class VolatilityPredictor:
    """波动率预测模型"""
    
    def __init__(self, model_type='mlp'):
        """
        初始化波动率预测模型
        
        参数:
            model_type (str): 模型类型，'mlp', 'lstm', 'gru', 'cnn'
        """
        self.model_type = model_type
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = 'implied_vol'
        self.history = None
    
    def preprocess_data(self, data, target_column=None, feature_columns=None):
        """
        预处理数据
        
        参数:
            data (DataFrame): 原始数据
            target_column (str): 目标列，默认为'implied_vol'
            feature_columns (list): 特征列，默认使用配置文件中的值
            
        返回:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 设置目标列和特征列
        self.target_column = target_column or self.target_column
        self.feature_columns = feature_columns or self.feature_columns
        
        # 检查数据中是否包含所有需要的列
        required_columns = self.feature_columns + [self.target_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少以下列: {missing_columns}")
        
        # 删除含有NaN的行
        data = data.dropna(subset=required_columns)
        
        # 分离特征和目标
        X = data[self.feature_columns].values
        y = data[self.target_column].values.reshape(-1, 1)
        
        # 特征标准化
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # 目标标准化
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y)
        
        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y_scaled, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=42
        )
        
        # 进一步分割为验证集和测试集
        val_size = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE), random_state=42
        )
        
        # 为LSTM和CNN模型调整数据形状
        if self.model_type in ['lstm', 'gru', 'cnn']:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_shape):
        """
        构建模型
        
        参数:
            input_shape (tuple): 输入形状
            
        返回:
            Model: Keras模型
        """
        if self.model_type == 'mlp':
            # 多层感知机
            model = Sequential()
            model.add(Input(shape=input_shape))
            
            # 添加隐藏层
            for units in NN_HIDDEN_LAYERS:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(0.2))
            
            # 输出层
            model.add(Dense(1))
            
        elif self.model_type == 'lstm':
            # LSTM模型
            model = Sequential()
            model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))
            
        elif self.model_type == 'gru':
            # GRU模型
            model = Sequential()
            model.add(GRU(64, input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(32))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))
            
        elif self.model_type == 'cnn':
            # CNN模型
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data, target_column=None, feature_columns=None, epochs=None, batch_size=None):
        """
        训练模型
        
        参数:
            data (DataFrame): 训练数据
            target_column (str): 目标列
            feature_columns (list): 特征列
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            
        返回:
            VolatilityPredictor: 自身实例
        """
        # 预处理数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(
            data, target_column, feature_columns
        )
        
        # 获取输入形状
        input_shape = X_train.shape[1] if self.model_type == 'mlp' else X_train.shape[1:]
        
        # 构建模型
        self.model = self.build_model(input_shape)
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs or TRAINING_EPOCHS,
            batch_size=batch_size or BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 评估模型
        self.evaluate(X_test, y_test)
        
        return self
    
    def predict(self, X):
        """
        预测波动率
        
        参数:
            X (DataFrame or ndarray): 特征数据
            
        返回:
            ndarray: 预测的波动率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 处理输入数据
        if isinstance(X, pd.DataFrame):
            if not all(col in X.columns for col in self.feature_columns):
                raise ValueError(f"输入数据缺少特征列: {[col for col in self.feature_columns if col not in X.columns]}")
            X = X[self.feature_columns].values
        
        # 标准化特征
        X_scaled = self.feature_scaler.transform(X)
        
        # 为LSTM和CNN模型调整数据形状
        if self.model_type in ['lstm', 'gru', 'cnn']:
            X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        # 预测
        y_pred_scaled = self.model.predict(X_scaled)
        
        # 逆变换回原始范围
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test (ndarray): 测试特征
            y_test (ndarray): 测试目标
            
        返回:
            dict: 评估指标
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred_scaled = self.model.predict(X_test)
        
        # 逆变换回原始范围
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test)
        
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 打印评估结果
        print(f"模型评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R^2: {r2:.6f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_training_history(self):
        """
        绘制训练历史
        
        返回:
            tuple: (fig, axes)
        """
        if self.history is None:
            raise ValueError("模型尚未训练")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失
        axes[0].plot(self.history.history['loss'], label='训练损失')
        axes[0].plot(self.history.history['val_loss'], label='验证损失')
        axes[0].set_xlabel('轮数')
        axes[0].set_ylabel('均方误差')
        axes[0].set_title('训练与验证损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制MAE
        axes[1].plot(self.history.history['mae'], label='训练MAE')
        axes[1].plot(self.history.history['val_mae'], label='验证MAE')
        axes[1].set_xlabel('轮数')
        axes[1].set_ylabel('平均绝对误差')
        axes[1].set_title('训练与验证MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_predictions(self, X_test, y_test, n_samples=50):
        """
        绘制预测结果
        
        参数:
            X_test (ndarray): 测试特征
            y_test (ndarray): 测试目标
            n_samples (int): 显示的样本数量
            
        返回:
            tuple: (fig, ax)
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred_scaled = self.model.predict(X_test[:n_samples])
        
        # 逆变换回原始范围
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test[:n_samples])
        
        # 绘制预测结果
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(y_true, 'b-', label='实际波动率')
        ax.plot(y_pred, 'r--', label='预测波动率')
        ax.set_xlabel('样本')
        ax.set_ylabel('隐含波动率')
        ax.set_title('波动率预测结果')
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def predict_vol_surface(self, vol_surface_data, future_days=5):
        """
        预测未来波动率曲面
        
        参数:
            vol_surface_data (DataFrame): 波动率曲面数据
            future_days (int): 预测未来天数
            
        返回:
            list: 预测的波动率曲面数据
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 确保数据包含所需的列
        required_cols = self.feature_columns + [self.target_column, 'strike', 'maturity']
        missing_cols = [col for col in required_cols if col not in vol_surface_data.columns]
        if missing_cols:
            raise ValueError(f"数据缺少以下列: {missing_cols}")
        
        # 复制原始数据
        current_data = vol_surface_data.copy()
        
        # 存储预测结果
        predicted_surfaces = []
        
        # 逐天预测
        for day in range(future_days):
            # 预测当前数据的波动率
            X = current_data[self.feature_columns]
            predicted_vols = self.predict(X)
            
            # 更新当前数据中的波动率
            current_data[self.target_column] = predicted_vols
            
            # 创建预测的波动率曲面数据
            surface_data = current_data[['strike', 'maturity', self.target_column]].copy()
            surface_data['day'] = day + 1
            
            predicted_surfaces.append(surface_data)
            
            # 更新特征以进行下一天的预测（例如，可以增加time_to_maturity）
            if 'time_to_maturity' in self.feature_columns:
                current_data['time_to_maturity'] = current_data['time_to_maturity'].apply(lambda x: max(0, x - 1/252))
        
        return predicted_surfaces
    
    def plot_predicted_vol_surface(self, predicted_surface, day=0, ax=None):
        """
        绘制预测的波动率曲面
        
        参数:
            predicted_surface (DataFrame): 预测的波动率曲面数据
            day (int): 要绘制的天数
            ax (matplotlib.axes.Axes): 坐标轴对象
            
        返回:
            matplotlib.axes.Axes: 坐标轴对象
        """
        # 筛选特定天的数据
        day_data = predicted_surface[predicted_surface['day'] == day + 1]
        
        # 创建网格
        strikes = np.sort(day_data['strike'].unique())
        maturities = np.sort(day_data['maturity'].unique())
        
        # 创建波动率网格
        vol_grid = np.zeros((len(strikes), len(maturities)))
        for i, k in enumerate(strikes):
            for j, t in enumerate(maturities):
                subset = day_data[(day_data['strike'] == k) & (day_data['maturity'] == t)]
                if len(subset) > 0:
                    vol_grid[i, j] = subset[self.target_column].values[0]
        
        # 创建绘图网格
        K, T = np.meshgrid(strikes, maturities)
        
        # 绘制曲面
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(K, T, vol_grid.T, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
        
        # 添加标签
        ax.set_xlabel('行权价 (K)')
        ax.set_ylabel('到期时间 (T)')
        ax.set_zlabel('隐含波动率 (\sigma)')
        ax.set_title(f'第{day+1}天预测波动率曲面')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return ax
    
    def save_model(self, path):
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 保存模型
        self.model.save(path)
        
        # 保存缩放器
        np.save(f"{path}_feature_scaler.npy", [
            self.feature_scaler.mean_,
            self.feature_scaler.scale_
        ])
        
        np.save(f"{path}_target_scaler.npy", [
            self.target_scaler.data_min_,
            self.target_scaler.data_max_,
            self.target_scaler.data_range_
        ])
        
        # 保存模型信息
        model_info = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        np.save(f"{path}_model_info.npy", model_info)
    
    @classmethod
    def load_model(cls, path):
        """
        加载模型
        
        参数:
            path (str): 模型路径
            
        返回:
            VolatilityPredictor: 加载的模型实例
        """
        # 加载模型信息
        model_info = np.load(f"{path}_model_info.npy", allow_pickle=True).item()
        
        # 创建实例
        instance = cls(model_type=model_info['model_type'])
        instance.feature_columns = model_info['feature_columns']
        instance.target_column = model_info['target_column']
        
        # 加载模型
        instance.model = tf.keras.models.load_model(path)
        
        # 加载特征缩放器
        feature_scaler_params = np.load(f"{path}_feature_scaler.npy", allow_pickle=True)
        instance.feature_scaler = StandardScaler()
        instance.feature_scaler.mean_ = feature_scaler_params[0]
        instance.feature_scaler.scale_ = feature_scaler_params[1]
        
        # 加载目标缩放器
        target_scaler_params = np.load(f"{path}_target_scaler.npy", allow_pickle=True)
        instance.target_scaler = MinMaxScaler()
        instance.target_scaler.data_min_ = target_scaler_params[0]
        instance.target_scaler.data_max_ = target_scaler_params[1]
        instance.target_scaler.data_range_ = target_scaler_params[2]
        
        return instance 