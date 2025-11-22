"""
Test Structural Break Analysis - Визуализация BOS для клиента
===============================================================

Анализирует последнюю неделю данных BTC и ETH на таймфреймах 1H и 4H.
Показывает Break of Structure (BOS) с примерами точек входа.

Генерирует HTML отчет с чистыми графиками и подробным описанием.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import seaborn as sns
import base64
from io import BytesIO

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10


class StructuralBreakAnalyzer:
    """
    Анализатор структурных сломов (Break of Structure).
    """
    
    def __init__(self, data: pd.DataFrame, lookback: int = 5):
        """
        Args:
            data: DataFrame с OHLC данными
            lookback: Период для определения swing points
        """
        self.data = data.copy()
        self.lookback = lookback
        self.swing_highs = []
        self.swing_lows = []
        self.bos_bullish = []
        self.bos_bearish = []
    
    def detect_swings(self):
        """
        Определить Swing High и Swing Low (БЕЗ look-ahead bias).
        """
        df = self.data
        n = len(df)
        lookback = self.lookback
        
        # Swing detection БЕЗ использования будущих данных
        for i in range(lookback * 2, n):
            swing_idx = i - lookback
            
            # Swing High
            is_swing_high = True
            for j in range(1, lookback + 1):
                if swing_idx - j < 0:
                    is_swing_high = False
                    break
                if df['high'].iloc[swing_idx] <= df['high'].iloc[swing_idx - j]:
                    is_swing_high = False
                    break
                if df['high'].iloc[swing_idx] <= df['high'].iloc[swing_idx + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                self.swing_highs.append({
                    'time': df.index[swing_idx],
                    'price': df['high'].iloc[swing_idx],
                    'index': swing_idx
                })
            
            # Swing Low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if swing_idx - j < 0:
                    is_swing_low = False
                    break
                if df['low'].iloc[swing_idx] >= df['low'].iloc[swing_idx - j]:
                    is_swing_low = False
                    break
                if df['low'].iloc[swing_idx] >= df['low'].iloc[swing_idx + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                self.swing_lows.append({
                    'time': df.index[swing_idx],
                    'price': df['low'].iloc[swing_idx],
                    'index': swing_idx
                })
    
    def detect_bos(self):
        """
        Определить Break of Structure (BOS).
        
        Бычий BOS: Цена пробивает последний Lower High медвежьего тренда
        Медвежий BOS: Цена пробивает последний Higher Low бычьего тренда
        """
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return
        
        df = self.data
        
        # Анализируем структуру
        for i in range(len(self.swing_highs) - 1):
            current_high = self.swing_highs[i + 1]
            prev_high = self.swing_highs[i]
            
            # Если текущий high выше предыдущего - возможен бычий BOS
            if current_high['price'] > prev_high['price']:
                # Это пробой LH (Lower High) медвежьего тренда
                # Ищем, где именно произошел пробой
                start_idx = prev_high['index']
                end_idx = current_high['index']
                
                # Находим момент пробоя
                for idx in range(start_idx, min(end_idx + 1, len(df))):
                    if df['close'].iloc[idx] > prev_high['price']:
                        self.bos_bullish.append({
                            'time': df.index[idx],
                            'price': df['close'].iloc[idx],
                            'broken_level': prev_high['price'],
                            'index': idx,
                            'context_start': prev_high['time'],
                            'context_end': current_high['time']
                        })
                        break
        
        for i in range(len(self.swing_lows) - 1):
            current_low = self.swing_lows[i + 1]
            prev_low = self.swing_lows[i]
            
            # Если текущий low ниже предыдущего - возможен медвежий BOS
            if current_low['price'] < prev_low['price']:
                # Это пробой HL (Higher Low) бычьего тренда
                start_idx = prev_low['index']
                end_idx = current_low['index']
                
                # Находим момент пробоя
                for idx in range(start_idx, min(end_idx + 1, len(df))):
                    if df['close'].iloc[idx] < prev_low['price']:
                        self.bos_bearish.append({
                            'time': df.index[idx],
                            'price': df['close'].iloc[idx],
                            'broken_level': prev_low['price'],
                            'index': idx,
                            'context_start': prev_low['time'],
                            'context_end': current_low['time']
                        })
                        break
    
    def analyze(self):
        """
        Запустить полный анализ.
        """
        self.detect_swings()
        self.detect_bos()
        
        return {
            'swing_highs': len(self.swing_highs),
            'swing_lows': len(self.swing_lows),
            'bos_bullish': len(self.bos_bullish),
            'bos_bearish': len(self.bos_bearish)
        }


def load_last_week_data(filepath: str) -> pd.DataFrame:
    """
    Загрузить последнюю неделю данных из CSV.
    """
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    
    # Подготовка данных
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Последняя неделя
    last_date = df.index.max()
    week_ago = last_date - timedelta(days=7)
    df_week = df[df.index >= week_ago].copy()
    
    print(f"Загружено {len(df_week)} свечей с {df_week.index.min()} по {df_week.index.max()}")
    
    return df_week


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Ресемплировать данные в нужный таймфрейм.
    """
    df_resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df_resampled


def plot_bos_analysis(df: pd.DataFrame, analyzer: StructuralBreakAnalyzer, 
                      symbol: str, timeframe: str):
    """
    Построить ЧИСТЫЙ график с анализом BOS (без текстовых наложений).
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 1. Основной график цены (свечи)
    for i in range(len(df)):
        color = '#26a69a' if df['close'].iloc[i] > df['open'].iloc[i] else '#ef5350'
        # Тени
        ax.plot([df.index[i], df.index[i]], 
               [df['low'].iloc[i], df['high'].iloc[i]], 
               color=color, linewidth=1.5, alpha=0.8, solid_capstyle='round')
        # Тело
        body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
        body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
        rect = plt.Rectangle((df.index[i], body_bottom), 
                            timedelta(minutes=0), body_height,
                            facecolor=color, edgecolor=color, 
                            alpha=0.9, linewidth=0, zorder=2)
        ax.add_patch(rect)
    
    # 2. Swing Highs (синие треугольники)
    for swing in analyzer.swing_highs:
        ax.plot(swing['time'], swing['price'], 'v', 
               markersize=12, color='#2196F3', 
               markeredgecolor='white', markeredgewidth=1.5,
               label='Swing High' if swing == analyzer.swing_highs[0] else '', zorder=3)
    
    # 3. Swing Lows (оранжевые треугольники)
    for swing in analyzer.swing_lows:
        ax.plot(swing['time'], swing['price'], '^', 
               markersize=12, color='#FF9800',
               markeredgecolor='white', markeredgewidth=1.5,
               label='Swing Low' if swing == analyzer.swing_lows[0] else '', zorder=3)
    
    # 4. Бычий BOS (зелёные отметки)
    for i, bos in enumerate(analyzer.bos_bullish):
        # Фоновая область
        ax.axvspan(bos['context_start'], bos['context_end'], 
                  alpha=0.1, color='#4CAF50', zorder=0)
        
        # Пробитый уровень
        time_fraction_start = (bos['context_start'] - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        time_fraction_end = (bos['time'] - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        ax.axhline(y=bos['broken_level'], 
                  xmin=time_fraction_start, xmax=time_fraction_end,
                  color='#4CAF50', linestyle='--', linewidth=2, alpha=0.6)
        
        # Точка BOS (звезда)
        ax.plot(bos['time'], bos['price'], '*', 
               markersize=25, color='#4CAF50', 
               markeredgecolor='white', markeredgewidth=2,
               label='БЫЧИЙ BOS' if i == 0 else '', zorder=5)
        
        # Точка входа
        entry_idx = min(bos['index'] + 3, len(df) - 1)
        entry_price = df['close'].iloc[entry_idx]
        ax.plot(df.index[entry_idx], entry_price, 'D', 
               markersize=14, color='#8BC34A', 
               markeredgecolor='white', markeredgewidth=2, zorder=6)
    
    # 5. Медвежий BOS (красные отметки)
    for i, bos in enumerate(analyzer.bos_bearish):
        # Фоновая область
        ax.axvspan(bos['context_start'], bos['context_end'], 
                  alpha=0.1, color='#F44336', zorder=0)
        
        # Пробитый уровень
        time_fraction_start = (bos['context_start'] - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        time_fraction_end = (bos['time'] - df.index[0]).total_seconds() / (df.index[-1] - df.index[0]).total_seconds()
        ax.axhline(y=bos['broken_level'], 
                  xmin=time_fraction_start, xmax=time_fraction_end,
                  color='#F44336', linestyle='--', linewidth=2, alpha=0.6)
        
        # Точка BOS (звезда)
        ax.plot(bos['time'], bos['price'], '*', 
               markersize=25, color='#F44336', 
               markeredgecolor='white', markeredgewidth=2,
               label='МЕДВЕЖИЙ BOS' if i == 0 else '', zorder=5)
        
        # Точка входа
        entry_idx = min(bos['index'] + 3, len(df) - 1)
        entry_price = df['close'].iloc[entry_idx]
        ax.plot(df.index[entry_idx], entry_price, 'D', 
               markersize=14, color='#EF9A9A', 
               markeredgecolor='white', markeredgewidth=2, zorder=6)
    
    # Настройка графика
    ax.set_title(f'{symbol} - {timeframe}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Время', fontsize=12, fontweight='bold')
    ax.set_ylabel('Цена ($)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig):
    """Конвертировать matplotlib фигуру в base64 для HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_html_report(results_data):
    """Сгенерировать HTML отчет с графиками и описанием."""
    
    html = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Break of Structure (BOS) Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            padding: 40px;
            background: #f8f9fa;
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        
        .chart-container img {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        
        .chart-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .stat-box {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .bullish { color: #4CAF50; }
        .bearish { color: #F44336; }
        
        .explanation {
            padding: 40px;
            background: white;
        }
        
        .explanation h2 {
            font-size: 2em;
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        
        .explanation-section {
            margin-bottom: 40px;
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
        }
        
        .explanation-section h3 {
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .explanation-section h3::before {
            content: '▶';
            color: #667eea;
        }
        
        .explanation-section p, .explanation-section ul {
            font-size: 1.1em;
            color: #444;
            margin-bottom: 15px;
        }
        
        .explanation-section ul {
            list-style-position: inside;
            padding-left: 20px;
        }
        
        .explanation-section li {
            margin-bottom: 10px;
            padding-left: 10px;
        }
        
        .key-point {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .key-point h3 {
            font-size: 1.8em;
            margin-bottom: 15px;
            color: white !important;
        }
        
        .key-point h3::before {
            content: '⭐';
            margin-right: 10px;
        }
        
        .legend {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 15px;
            background: white;
            padding: 15px;
            border-radius: 10px;
        }
        
        .legend-icon {
            font-size: 2em;
        }
        
        .legend-text {
            flex: 1;
        }
        
        .legend-label {
            font-weight: bold;
            font-size: 1.1em;
            color: #333;
        }
        
        .legend-description {
            font-size: 0.95em;
            color: #666;
        }
        
        .trading-steps {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .trading-steps h4 {
            color: #856404;
            font-size: 1.3em;
            margin-bottom: 15px;
        }
        
        .trading-steps ol {
            padding-left: 25px;
        }
        
        .trading-steps li {
            margin-bottom: 10px;
            color: #856404;
            font-weight: 500;
        }
        
        .warning-box {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .warning-box h4 {
            color: #721c24;
            font-size: 1.3em;
            margin-bottom: 15px;
        }
        
        .warning-box ul {
            padding-left: 25px;
        }
        
        .warning-box li {
            color: #721c24;
            margin-bottom: 10px;
        }
        
        footer {
            background: #333;
            color: white;
            text-align: center;
            padding: 30px;
            font-size: 0.95em;
        }
        
        @media (max-width: 1024px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .legend {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Break of Structure (BOS) Analysis</h1>
            <p>Анализ структурных сломов рынка • Последняя неделя</p>
            <p style="font-size: 0.9em; margin-top: 10px;">BTC & ETH • Таймфреймы: 1H и 4H</p>
        </header>
        
        <div class="charts-grid">
"""
    
    # Добавить графики
    for result in results_data:
        html += f"""
            <div class="chart-container">
                <div class="chart-title">{result['symbol']} - {result['timeframe']}</div>
                <img src="data:image/png;base64,{result['image']}" alt="{result['symbol']} {result['timeframe']}">
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-label">Swing High</div>
                        <div class="stat-value">{result['stats']['swing_highs']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Swing Low</div>
                        <div class="stat-value">{result['stats']['swing_lows']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label bullish">Бычий BOS</div>
                        <div class="stat-value bullish">{result['stats']['bos_bullish']}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label bearish">Медвежий BOS</div>
                        <div class="stat-value bearish">{result['stats']['bos_bearish']}</div>
                    </div>
                </div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="explanation">
            <h2>Что такое Break of Structure (BOS)?</h2>
            
            <div class="key-point">
                <h3>BOS - это ТОЧКА или ОБЛАСТЬ?</h3>
                <p><strong>Ответ: BOS - это ТОЧКА (момент) во времени</strong>, когда цена пробивает ключевой уровень структуры.</p>
                <p><strong>Аналогия со светофором:</strong> Переключение с красного на зелёный происходит в КОНКРЕТНЫЙ момент, но мы видим подготовку (желтый) и последствия (машины поехали).</p>
                <p style="margin-top: 15px;"><strong>На графике:</strong> Мы показываем светлую ОБЛАСТЬ для контекста, но ⭐ звезда обозначает точный МОМЕНТ пробоя.</p>
            </div>
            
            <div class="explanation-section">
                <h3>Элементы на графике</h3>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-icon">🔵▼</div>
                        <div class="legend-text">
                            <div class="legend-label">Swing High</div>
                            <div class="legend-description">Локальный максимум (вершина)</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon">🟠▲</div>
                        <div class="legend-text">
                            <div class="legend-label">Swing Low</div>
                            <div class="legend-description">Локальный минимум (дно)</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon">🟢⭐</div>
                        <div class="legend-text">
                            <div class="legend-label">Бычий BOS</div>
                            <div class="legend-description">Момент пробоя вверх - сигнал к покупке</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon">🔴⭐</div>
                        <div class="legend-text">
                            <div class="legend-label">Медвежий BOS</div>
                            <div class="legend-description">Момент пробоя вниз - сигнал к продаже</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon">💎</div>
                        <div class="legend-text">
                            <div class="legend-label">Точка входа</div>
                            <div class="legend-description">Вход в сделку после подтверждения BOS</div>
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon">---</div>
                        <div class="legend-text">
                            <div class="legend-label">Пунктирная линия</div>
                            <div class="legend-description">Пробитый уровень структуры</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="explanation-section">
                <h3 style="color: #4CAF50;">Бычий BOS (Bullish Break of Structure)</h3>
                <p><strong>Что происходит:</strong></p>
                <ul>
                    <li>Рынок находится в нисходящем тренде (серия LH и LL - Lower Highs и Lower Lows)</li>
                    <li>Цена формирует новый минимум</li>
                    <li>⭐ Цена пробивает ВВЕРХ последний Lower High (LH) медвежьего тренда</li>
                    <li>Медвежья структура СЛОМАНА</li>
                    <li>💎 Сигнал к ПОКУПКЕ (LONG)</li>
                </ul>
                <p style="margin-top: 15px;"><strong>Почему это важно:</strong></p>
                <ul>
                    <li>"Умные деньги" (крупные игроки) начали покупать</li>
                    <li>У них достаточно силы для слома нисходящего тренда</li>
                    <li>Высокая вероятность продолжения роста</li>
                </ul>
            </div>
            
            <div class="explanation-section">
                <h3 style="color: #F44336;">Медвежий BOS (Bearish Break of Structure)</h3>
                <p><strong>Что происходит:</strong></p>
                <ul>
                    <li>Рынок находится в восходящем тренде (серия HH и HL - Higher Highs и Higher Lows)</li>
                    <li>Цена формирует новый максимум</li>
                    <li>⭐ Цена пробивает ВНИЗ последний Higher Low (HL) бычьего тренда</li>
                    <li>Бычья структура СЛОМАНА</li>
                    <li>💎 Сигнал к ПРОДАЖЕ (SHORT)</li>
                </ul>
                <p style="margin-top: 15px;"><strong>Почему это важно:</strong></p>
                <ul>
                    <li>"Умные деньги" начали продавать</li>
                    <li>У них достаточно силы для слома восходящего тренда</li>
                    <li>Высокая вероятность продолжения падения</li>
                </ul>
            </div>
            
            <div class="explanation-section">
                <h3>Как торговать по BOS</h3>
                
                <div class="trading-steps">
                    <h4>После БЫЧЬЕГО BOS (покупка):</h4>
                    <ol>
                        <li>Дождаться подтверждения - 2-3 свечи после пробоя</li>
                        <li>Войти в LONG (купить) в точке 💎</li>
                        <li>Поставить стоп-лосс НИЖЕ последнего Swing Low</li>
                        <li>Цель прибыли - следующий Swing High</li>
                        <li>Соотношение риск/прибыль минимум 1:2</li>
                    </ol>
                </div>
                
                <div class="trading-steps">
                    <h4>После МЕДВЕЖЬЕГО BOS (продажа):</h4>
                    <ol>
                        <li>Дождаться подтверждения - 2-3 свечи после пробоя</li>
                        <li>Войти в SHORT (продать) в точке 💎</li>
                        <li>Поставить стоп-лосс ВЫШЕ последнего Swing High</li>
                        <li>Цель прибыли - следующий Swing Low</li>
                        <li>Соотношение риск/прибыль минимум 1:2</li>
                    </ol>
                </div>
            </div>
            
            <div class="warning-box">
                <h4>⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ</h4>
                <ul>
                    <li>BOS - это НЕ гарантия разворота, а СИГНАЛ о смене структуры</li>
                    <li>ВСЕГДА используйте стоп-лосс для защиты капитала</li>
                    <li>НЕ входите сразу в момент пробоя - дождитесь подтверждения</li>
                    <li>BOS лучше работает на высоких таймфреймах (4H, Daily)</li>
                    <li>Учитывайте общий рыночный контекст и новости</li>
                    <li>Рискуйте не более 1-2% капитала на одну сделку</li>
                </ul>
            </div>
        </div>
        
        <footer>
            <p><strong>Smart Money Concepts Trading System</strong></p>
            <p style="margin-top: 10px;">Анализ создан автоматически • Данные: последняя неделя</p>
            <p style="margin-top: 5px; opacity: 0.7;">Не является инвестиционной рекомендацией</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html


def main():
    """
    Главная функция анализа - генерирует HTML отчет.
    """
    print("\n" + "="*90)
    print(" АНАЛИЗ СТРУКТУРНЫХ СЛОМОВ (BREAK OF STRUCTURE)")
    print("="*90 + "\n")
    
    # Файлы данных
    files = {
        'BTC': '/home/edward/Documents/smart money/data/BTCUSDT_1m_binance.csv',
        'ETH': '/home/edward/Documents/smart money/data/ETHUSDT_1m_binance.csv'
    }
    
    # Таймфреймы для анализа
    timeframes = {
        '1H': '1h',
        '4H': '4h'
    }
    
    results_data = []
    
    for symbol, filepath in files.items():
        # Загрузить данные
        df_1m = load_last_week_data(filepath)
        
        for tf_name, tf_code in timeframes.items():
            print(f"\n{'='*80}")
            print(f"Анализ {symbol} на {tf_name}")
            print('='*80)
            
            # Ресемплировать
            df_tf = resample_to_timeframe(df_1m, tf_code)
            print(f"Данные после ресемплинга: {len(df_tf)} свечей")
            
            # Анализ
            analyzer = StructuralBreakAnalyzer(df_tf, lookback=5)
            stats = analyzer.analyze()
            
            print(f"\nРезультаты анализа:")
            print(f"  Swing Highs: {stats['swing_highs']}")
            print(f"  Swing Lows: {stats['swing_lows']}")
            print(f"  🟢 Бычий BOS: {stats['bos_bullish']}")
            print(f"  🔴 Медвежий BOS: {stats['bos_bearish']}")
            
            # Детали BOS
            if analyzer.bos_bullish:
                print(f"\n  Бычий BOS найден:")
                for i, bos in enumerate(analyzer.bos_bullish, 1):
                    print(f"    #{i}: {bos['time']} по цене ${bos['price']:.2f}")
                    print(f"        Пробит уровень: ${bos['broken_level']:.2f}")
            
            if analyzer.bos_bearish:
                print(f"\n  Медвежий BOS найден:")
                for i, bos in enumerate(analyzer.bos_bearish, 1):
                    print(f"    #{i}: {bos['time']} по цене ${bos['price']:.2f}")
                    print(f"        Пробит уровень: ${bos['broken_level']:.2f}")
            
            # Создать график
            print(f"Создание графика для {symbol} {tf_name}...")
            fig = plot_bos_analysis(df_tf, analyzer, symbol, tf_name)
            
            # Конвертировать в base64
            img_base64 = fig_to_base64(fig)
            
            # Сохранить результаты
            results_data.append({
                'symbol': symbol,
                'timeframe': tf_name,
                'image': img_base64,
                'stats': stats
            })
    
    # Генерация HTML
    print(f"\n{'='*90}")
    print("Генерация HTML отчета...")
    html_content = generate_html_report(results_data)
    
    # Сохранить HTML
    html_filename = 'structural_break_analysis.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML отчет сохранён: {html_filename}")
    print(f"{'='*90}\n")
    
    # Финальное объяснение
    print("\n" + "="*90)
    print(" ОТВЕТ НА ВОПРОС: BOS - ЭТО ТОЧКА ИЛИ ОБЛАСТЬ?")
    print("="*90)
    print("""
BOS (Break of Structure) - это ТОЧКА (момент) во времени, когда происходит пробой.

Аналогия со светофором:
   • Красный → Зелёный: Переключение происходит в КОНКРЕТНЫЙ момент
   • НО мы видим подготовку (жёлтый) и последствия (машины едут)

На графике:
   • ⭐ Звезда - ТОЧКА пробоя (момент BOS)
   • Светлый фон - ОБЛАСТЬ контекста (что было до и после)
   • Мы показываем область для ПОНИМАНИЯ, но BOS - это конкретный момент

Для торговли важен именно МОМЕНТ пробоя, но мы смотрим на КОНТЕКСТ 
для подтверждения и понимания силы движения.

Откройте файл structural_break_analysis.html в браузере для просмотра!
    """)
    print("="*90 + "\n")


if __name__ == '__main__':
    main()
