import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, Decimal, TAindicators, IntParameter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import talib.abstract as ta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIStrategy(IStrategy):
    """
    Estrategia avanzada con IA para obtener +10% de ganancia por operación
    Combinación de modelo de machine learning con indicadores técnicos
    """
    
    # Parámetros optimizables
    buy_rsi = IntParameter(25, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 75, default=70, space="sell")
    buy_volume = IntParameter(1, 5, default=2, space="buy")
    
    # Timeframe
    timeframe = '15m'
    
    # Stop Loss
    stoploss = -0.05  # -5%
    
    # ROI con objetivo del 10%
    minimal_roi = {
        "0": 0.10,  # 10% de ganancia
        "10": 0.05,
        "20": 0.02,
        "30": 0.01
    }
    
    # Trailing Stop
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True
    
    # Hyperopt
    buy_protection_params = {
        "cooldown_lookback": IntParameter(2, 48, default=5, space="buy"),
        "stoploss_duration": IntParameter(12, 200, default=50, space="buy"),
    }
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
        self.load_ai_model()
        
    def load_ai_model(self):
        """Cargar o entrenar el modelo de IA"""
        try:
            self.model = joblib.load('ai_trading_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.logger.info("Modelo de IA cargado exitosamente")
        except:
            self.logger.info("No se encontró modelo preentrenado, se entrenará uno nuevo")
            self.model = self.train_ai_model()
    
    def extract_features(self, dataframe: DataFrame) -> DataFrame:
        """Extraer características para el modelo de IA"""
        df = dataframe.copy()
        
        # Indicadores técnicos
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['macd'] = ta.MACD(df)['macd']
        df['adx'] = ta.ADX(df)
        df['plus_di'] = ta.PLUS_DI(df)
        df['minus_di'] = ta.MINUS_DI(df)
        df['williams'] = ta.WILLR(df, timeperiod=14)
        df['uo'] = ta.ULTOSC(df)
        df['atr'] = ta.ATR(df, timeperiod=14)
        
        # Medias móviles
        df['sma_5'] = ta.SMA(df, timeperiod=5)
        df['sma_10'] = ta.SMA(df, timeperiod=10)
        df['sma_20'] = ta.SMA(df, timeperiod=20)
        df['sma_50'] = ta.SMA(df, timeperiod=50)
        df['ema_5'] = ta.EMA(df, timeperiod=5)
        df['ema_10'] = ta.EMA(df, timeperiod=10)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df['bb_upper'] = bollinger['upperband']
        df['bb_middle'] = bollinger['middleband']
        df['bb_lower'] = bollinger['lowerband']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volumen
        df['volume_ma'] = ta.SMA(df, timeperiod=20, price='volume')
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum
        df['momentum'] = ta.MOM(df, timeperiod=10)
        df['rate_of_change'] = ta.ROC(df, timeperiod=10)
        
        # Patrones de velas
        df['doji'] = ta.CDLDOJI(df)
        df['hammer'] = ta.CDLHAMMER(df)
        df['engulfing'] = ta.CDLENGULFING(df)
        
        return df
    
    def create_target(self, dataframe: DataFrame, periods: int = 3) -> Series:
        """Crear variable objetivo: 1 si el precio sube un 10% en los próximos 'periods' períodos"""
        future_return = dataframe['close'].pct_change(periods).shift(-periods)
        target = (future_return >= 0.10).astype(int)  # 10% de ganancia
        return target
    
    def train_ai_model(self):
        """Entrenar modelo de IA con datos históricos"""
        from freqtrade.data.history import load_data
        
        # Cargar datos históricos
        data = load_data(
            datadir=Path('user_data/data/binance'),
            pairs=['BTC/USDT'],
            timeframe='15m',
            timerange='20230101-20231231'
        )
        
        # Extraer características
        features_df = self.extract_features(data['BTC/USDT'])
        
        # Crear variable objetivo
        target = self.create_target(features_df, periods=4)
        
        # Preparar datos para entrenamiento
        feature_columns = [
            'rsi', 'macd', 'adx', 'plus_di', 'minus_di', 'williams', 'uo', 'atr',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'volume_ma', 'volume_ratio', 'momentum', 'rate_of_change',
            'doji', 'hammer', 'engulfing'
        ]
        
        # Eliminar filas con NaN
        valid_indices = features_df[feature_columns].dropna().index
        X = features_df.loc[valid_indices, feature_columns]
        y = target.loc[valid_indices]
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo (Gradient Boosting)
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Precisión del modelo: {accuracy:.2f}")
        
        # Guardar modelo
        joblib.dump(model, 'ai_trading_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        return model
    
    def get_ai_signal(self, dataframe: DataFrame) -> Series:
        """Obtener señal de compra del modelo de IA"""
        # Extraer características
        features_df = self.extract_features(dataframe)
        
        # Columnas de características
        feature_columns = [
            'rsi', 'macd', 'adx', 'plus_di', 'minus_di', 'williams', 'uo', 'atr',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'volume_ma', 'volume_ratio', 'momentum', 'rate_of_change',
            'doji', 'hammer', 'engulfing'
        ]
        
        # Obtener la última vela
        latest_data = features_df[feature_columns].iloc[-1:].copy()
        
        # Escalar datos
        latest_scaled = self.scaler.transform(latest_data)
        
        # Predecir
        prediction = self.model.predict(latest_scaled)
        probability = self.model.predict_proba(latest_scaled)[0][1]
        
        return prediction[0], probability
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicadores básicos
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['sma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Volume
        dataframe['volume_ma'] = ta.SMA(dataframe, timeperiod=20, price='volume')
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        # Señal de IA
        ai_signal, ai_prob = self.get_ai_signal(dataframe)
        conditions.append(ai_signal == 1)
        conditions.append(ai_prob > 0.7)  # Probabilidad mínima del 70%
        
        # Condiciones técnicas adicionales
        conditions.append(dataframe['rsi'] < self.buy_rsi.value)
        conditions.append(dataframe['volume_ratio'] > self.buy_volume.value)
        conditions.append(dataframe['close'] > dataframe['sma_20'])
        conditions.append(dataframe['sma_20'] > dataframe['sma_50'])
        conditions.append(dataframe['adx'] > 25)  # Tendencia fuerte
        
        # Aplicar condiciones
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        # Señal de venta de IA
        ai_signal, ai_prob = self.get_ai_signal(dataframe)
        conditions.append(ai_signal == 0)
        
        # Condiciones técnicas
        conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        conditions.append(dataframe['close'] < dataframe['sma_20'])
        
        # Aplicar condiciones
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'] = 1
        
        return dataframe
