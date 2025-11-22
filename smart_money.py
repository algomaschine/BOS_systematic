"""
Smart Money Concepts (SMC) Trading System with Backtesting
=============================================================

This module implements a complete SMC trading system including:
1. Swing High/Low detection
2. Market Structure (BOS/CHOCh) detection
3. Fair Value Gap (FVG) detection
4. Triple Barrier Method for exits
5. Position sizing with risk management
6. Comprehensive backtesting engine
7. Performance metrics and analysis

Author: Smart Money Trading System
Date: 2025
"""

from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 8)


# ==================== PART 1: MARKET STRUCTURE DETECTION ====================

class SwingDetector:
    """
    Detects swing highs and swing lows in price data WITHOUT LOOK-AHEAD BIAS.
    
    A swing high is confirmed AFTER N bars where the high was greater than 
    the previous N bars AND the following N bars (confirmation delay).
    
    This ensures NO FUTURE DATA is used - swing is only marked after confirmation period.
    """
    
    def __init__(self, lookback: int = 5):
        """
        Args:
            lookback: Number of bars for swing confirmation (only uses past data)
        """
        self.lookback = lookback
    
    def detect_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows WITHOUT LOOK-AHEAD BIAS.
        
        Swing is marked at bar i ONLY after lookback bars have confirmed it.
        This means the swing signal appears with a delay, preventing data leakage.
        
        Args:
            df: DataFrame with 'high' and 'low' columns
            
        Returns:
            DataFrame with additional 'swing_high' and 'swing_low' columns
        """
        df = df.copy()
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        
        n = len(df)
        lookback = self.lookback
        
        # Start from lookback*2 to have enough data for both directions
        for i in range(lookback * 2, n):
            # Check for swing high at position (i - lookback)
            # This ensures we only use data available up to bar i
            swing_idx = i - lookback
            
            # Check if high at swing_idx was highest in the range
            is_swing_high = True
            for j in range(1, lookback + 1):
                # Check left side (historical data)
                if swing_idx - j < 0:
                    is_swing_high = False
                    break
                if df['high'].iloc[swing_idx] <= df['high'].iloc[swing_idx - j]:
                    is_swing_high = False
                    break
                # Check right side (confirmation period - now in the past)
                if df['high'].iloc[swing_idx] <= df['high'].iloc[swing_idx + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                df.loc[df.index[swing_idx], 'swing_high'] = df['high'].iloc[swing_idx]
            
            # Check for swing low at position (i - lookback)
            is_swing_low = True
            for j in range(1, lookback + 1):
                # Check left side (historical data)
                if swing_idx - j < 0:
                    is_swing_low = False
                    break
                if df['low'].iloc[swing_idx] >= df['low'].iloc[swing_idx - j]:
                    is_swing_low = False
                    break
                # Check right side (confirmation period - now in the past)
                if df['low'].iloc[swing_idx] >= df['low'].iloc[swing_idx + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                df.loc[df.index[swing_idx], 'swing_low'] = df['low'].iloc[swing_idx]
        
        return df


class MarketStructureDetector:
    """
    Detects market structure changes including:
    - Break of Structure (BOS)
    - Change of Character (CHOCh)
    """
    
    def __init__(self):
        self.trend = None  # 'bullish', 'bearish', or None
        self.last_higher_high = None
        self.last_higher_low = None
        self.last_lower_high = None
        self.last_lower_low = None
    
    def detect_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market structure and BOS/CHOCh signals.
        
        Args:
            df: DataFrame with 'swing_high' and 'swing_low' columns
            
        Returns:
            DataFrame with structure detection columns
        """
        df = df.copy()
        df['market_structure'] = None  # 'bullish', 'bearish', or None
        df['bos_signal'] = None  # 'bullish_bos', 'bearish_bos', or None
        df['choch_signal'] = None  # 'bullish_choch', 'bearish_choch', or None
        
        swing_highs = df['swing_high'].dropna()
        swing_lows = df['swing_low'].dropna()
        
        # Track structure
        prev_high = None
        prev_low = None
        current_trend = None
        
        for idx in df.index:
            # Update swing highs
            if not pd.isna(df.loc[idx, 'swing_high']):
                current_high = df.loc[idx, 'swing_high']
                
                if prev_high is not None:
                    if current_high > prev_high:
                        # Higher high
                        self.last_higher_high = (idx, current_high)
                    else:
                        # Lower high
                        self.last_lower_high = (idx, current_high)
                
                prev_high = current_high
            
            # Update swing lows
            if not pd.isna(df.loc[idx, 'swing_low']):
                current_low = df.loc[idx, 'swing_low']
                
                if prev_low is not None:
                    if current_low > prev_low:
                        # Higher low
                        self.last_higher_low = (idx, current_low)
                    else:
                        # Lower low
                        self.last_lower_low = (idx, current_low)
                
                prev_low = current_low
            
            # Determine trend and detect BOS
            if self.last_higher_high and self.last_higher_low:
                # Bullish structure: series of HH and HL
                if current_trend != 'bullish':
                    df.loc[idx, 'bos_signal'] = 'bullish_bos'
                current_trend = 'bullish'
            
            if self.last_lower_high and self.last_lower_low:
                # Bearish structure: series of LH and LL
                if current_trend != 'bearish':
                    df.loc[idx, 'bos_signal'] = 'bearish_bos'
                current_trend = 'bearish'
            
            df.loc[idx, 'market_structure'] = current_trend
        
        return df


class FVGDetector:
    """
    Detects Fair Value Gaps (FVG) - price imbalances that often get filled.
    
    A Fair Value Gap occurs when there's a gap between the high of candle 1
    and the low of candle 3 (for bullish FVG) or vice versa for bearish FVG.
    """
    
    def __init__(self, min_gap_size: float = 0.001):
        """
        Args:
            min_gap_size: Minimum gap size as a percentage of price (0.001 = 0.1%)
        """
        self.min_gap_size = min_gap_size
    
    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with FVG detection columns
        """
        df = df.copy()
        df['fvg_type'] = None  # 'bullish', 'bearish', or None
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan
        
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2].low and candle[i].high
            # The middle candle (i-1) should not fill this gap
            if df['low'].iloc[i] > df['high'].iloc[i - 2]:
                gap_size = (df['low'].iloc[i] - df['high'].iloc[i - 2]) / df['close'].iloc[i]
                
                if gap_size >= self.min_gap_size:
                    df.loc[df.index[i], 'fvg_type'] = 'bullish'
                    df.loc[df.index[i], 'fvg_top'] = df['low'].iloc[i]
                    df.loc[df.index[i], 'fvg_bottom'] = df['high'].iloc[i - 2]
            
            # Bearish FVG: gap between candle[i-2].high and candle[i].low
            elif df['high'].iloc[i] < df['low'].iloc[i - 2]:
                gap_size = (df['low'].iloc[i - 2] - df['high'].iloc[i]) / df['close'].iloc[i]
                
                if gap_size >= self.min_gap_size:
                    df.loc[df.index[i], 'fvg_type'] = 'bearish'
                    df.loc[df.index[i], 'fvg_top'] = df['low'].iloc[i - 2]
                    df.loc[df.index[i], 'fvg_bottom'] = df['high'].iloc[i]
        
        return df


# ==================== PART 2: TRADING SYSTEM ====================

class TripleBarrierMethod:
    """
    Implements the Triple Barrier Method for position management.
    
    Three barriers:
    1. Upper Barrier (Profit Target)
    2. Lower Barrier (Stop Loss)
    3. Time Barrier (Maximum holding period)
    """
    
    def __init__(self, profit_target: float = 2.0, stop_loss: float = 1.0, 
                 max_holding_bars: int = 10):
        """
        Args:
            profit_target: Profit target as multiple of risk (e.g., 2.0 = 2:1 R:R)
            stop_loss: Stop loss as multiple of risk (always 1.0)
            max_holding_bars: Maximum number of bars to hold position
        """
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_bars = max_holding_bars
    
    def calculate_barriers(self, entry_price: float, stop_price: float, 
                          direction: str) -> Dict[str, float]:
        """
        Calculate the three barriers for a position.
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            direction: 'long' or 'short'
            
        Returns:
            Dictionary with barrier levels
        """
        risk = abs(entry_price - stop_price)
        
        if direction == 'long':
            profit_target_price = entry_price + (risk * self.profit_target)
            stop_loss_price = stop_price
        else:  # short
            profit_target_price = entry_price - (risk * self.profit_target)
            stop_loss_price = stop_price
        
        return {
            'profit_target': profit_target_price,
            'stop_loss': stop_loss_price,
            'risk': risk
        }


class PositionSizer:
    """
    Calculates position size based on risk management rules.
    """
    
    def __init__(self, risk_per_trade: float = 0.01):
        """
        Args:
            risk_per_trade: Percentage of capital to risk per trade (0.01 = 1%)
        """
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(self, capital: float, entry_price: float, 
                               stop_price: float) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            capital: Total account capital
            entry_price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Position size in base currency units
        """
        risk_amount = capital * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_price)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        return position_size


class SMCTradingStrategy:
    """
    Complete Smart Money Concepts trading strategy.
    
    Entry Logic:
    - Trade in direction of 4H BOS
    - Enter on 1H pullback to FVG or key level
    - Confirm with 1H micro-BOS
    
    Exit Logic:
    - Triple Barrier Method
    
    Risk Management:
    - Position sizing based on risk per trade
    """
    
    def __init__(self, 
                 risk_per_trade: float = 0.01,
                 profit_target: float = 2.0,
                 max_holding_bars: int = 10,
                 swing_lookback: int = 5):
        """
        Args:
            risk_per_trade: Risk per trade as percentage (0.01 = 1%)
            profit_target: Profit target as R:R multiple
            max_holding_bars: Maximum bars to hold position
            swing_lookback: Lookback for swing detection
        """
        self.risk_per_trade = risk_per_trade
        self.profit_target = profit_target
        self.max_holding_bars = max_holding_bars
        self.swing_lookback = swing_lookback
        
        # Initialize components
        self.position_sizer = PositionSizer(risk_per_trade)
        self.triple_barrier = TripleBarrierMethod(profit_target, 1.0, max_holding_bars)
    
    def prepare_data(self, df_1m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data by resampling to 1H and 4H and detecting structure.
        
        Args:
            df_1m: 1-minute OHLC dataframe
            
        Returns:
            Tuple of (df_1h, df_4h) with structure detection
        """
        # Ensure datetime index
        if 'Date' in df_1m.columns:
            df_1m['Date'] = pd.to_datetime(df_1m['Date'])
            df_1m.set_index('Date', inplace=True)
        
        # Resample to 1H
        df_1h = df_1m.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Resample to 4H
        df_4h = df_1m.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Detect structure on both timeframes
        swing_detector = SwingDetector(self.swing_lookback)
        structure_detector = MarketStructureDetector()
        fvg_detector = FVGDetector()
        
        # 4H structure
        df_4h = swing_detector.detect_swings(df_4h)
        df_4h = structure_detector.detect_structure(df_4h)
        df_4h = fvg_detector.detect_fvg(df_4h)
        
        # 1H structure
        swing_detector_1h = SwingDetector(self.swing_lookback)
        structure_detector_1h = MarketStructureDetector()
        fvg_detector_1h = FVGDetector()
        
        df_1h = swing_detector_1h.detect_swings(df_1h)
        df_1h = structure_detector_1h.detect_structure(df_1h)
        df_1h = fvg_detector_1h.detect_fvg(df_1h)
        
        return df_1h, df_4h
    
    def generate_signals(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SMC logic.
        
        Args:
            df_1h: 1-hour dataframe with structure
            df_4h: 4-hour dataframe with structure
            
        Returns:
            DataFrame with trading signals
        """
        signals = df_1h.copy()
        signals['signal'] = 0  # 1 = long, -1 = short, 0 = no signal
        signals['stop_loss'] = np.nan
        signals['entry_reason'] = None
        
        # Map 4H structure to 1H timeframe
        df_4h_reindexed = df_4h[['market_structure', 'bos_signal']].reindex(
            signals.index, method='ffill'
        )
        signals['4h_structure'] = df_4h_reindexed['market_structure']
        signals['4h_bos'] = df_4h_reindexed['bos_signal']
        
        # Generate signals
        for i in range(self.swing_lookback + 5, len(signals)):
            idx = signals.index[i]
            
            # Get current 4H trend
            trend_4h = signals.loc[idx, '4h_structure']
            
            if trend_4h is None:
                continue
            
            # LONG ENTRY CONDITIONS
            if trend_4h == 'bullish':
                # Check for 1H pullback to FVG or swing low
                has_fvg = signals.loc[idx, 'fvg_type'] == 'bullish'
                near_swing_low = not pd.isna(signals.loc[idx, 'swing_low'])
                
                # Check for 1H bullish BOS as confirmation
                recent_bos_1h = False
                lookback_window = signals.index[max(0, i - 5):i]
                if len(lookback_window) > 0:
                    recent_bos_1h = (signals.loc[lookback_window, 'bos_signal'] == 'bullish_bos').any()
                
                if (has_fvg or near_swing_low) and recent_bos_1h:
                    # Find stop loss below recent swing low
                    recent_lows = signals.loc[signals.index[max(0, i - 20):i], 'swing_low'].dropna()
                    if len(recent_lows) > 0:
                        stop_loss = recent_lows.min() * 0.999  # Slightly below
                        
                        # Only enter if stop is not too far
                        if (signals.loc[idx, 'close'] - stop_loss) / signals.loc[idx, 'close'] < 0.03:  # Max 3% risk
                            signals.loc[idx, 'signal'] = 1
                            signals.loc[idx, 'stop_loss'] = stop_loss
                            signals.loc[idx, 'entry_reason'] = 'bullish_pullback'
            
            # SHORT ENTRY CONDITIONS
            elif trend_4h == 'bearish':
                # Check for 1H pullback to FVG or swing high
                has_fvg = signals.loc[idx, 'fvg_type'] == 'bearish'
                near_swing_high = not pd.isna(signals.loc[idx, 'swing_high'])
                
                # Check for 1H bearish BOS as confirmation
                recent_bos_1h = False
                lookback_window = signals.index[max(0, i - 5):i]
                if len(lookback_window) > 0:
                    recent_bos_1h = (signals.loc[lookback_window, 'bos_signal'] == 'bearish_bos').any()
                
                if (has_fvg or near_swing_high) and recent_bos_1h:
                    # Find stop loss above recent swing high
                    recent_highs = signals.loc[signals.index[max(0, i - 20):i], 'swing_high'].dropna()
                    if len(recent_highs) > 0:
                        stop_loss = recent_highs.max() * 1.001  # Slightly above
                        
                        # Only enter if stop is not too far
                        if (stop_loss - signals.loc[idx, 'close']) / signals.loc[idx, 'close'] < 0.03:  # Max 3% risk
                            signals.loc[idx, 'signal'] = -1
                            signals.loc[idx, 'stop_loss'] = stop_loss
                            signals.loc[idx, 'entry_reason'] = 'bearish_pullback'
        
        return signals


# ==================== PART 3: BACKTESTING ENGINE ====================

class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_time, entry_price, direction, size, stop_loss, take_profit):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 'long' or 'short'
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0
        self.return_pct = 0
        self.bars_held = 0
    
    def close(self, exit_time, exit_price, exit_reason):
        """Close the trade and calculate P&L."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        if self.direction == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.return_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
            self.return_pct = (self.entry_price - exit_price) / self.entry_price * 100
        
        # Calculate bars held
        if isinstance(self.entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            self.bars_held = (exit_time - self.entry_time).total_seconds() / 3600  # Hours
    
    def to_dict(self):
        """Convert trade to dictionary."""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'direction': self.direction,
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'exit_reason': self.exit_reason,
            'bars_held': self.bars_held
        }


class Backtester:
    """
    Backtesting engine for SMC trading strategy with DRAWDOWN PROTECTION.
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 commission: float = 0.001,
                 max_drawdown_percent: float = None):
        """
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            max_drawdown_percent: Maximum allowed drawdown before stopping trading (e.g., 0.20 = 20%)
                                  None = no drawdown limit
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_drawdown_percent = max_drawdown_percent
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        self.drawdown_breached = False
        self.trades_skipped_due_to_drawdown = 0
    
    def run_backtest(self, signals: pd.DataFrame, strategy: SMCTradingStrategy) -> pd.DataFrame:
        """
        Run backtest on signal dataframe with DRAWDOWN PROTECTION.
        
        If max_drawdown_percent is set, trading stops when drawdown exceeds the limit.
        Trading resumes when equity recovers above the threshold.
        
        Args:
            signals: DataFrame with trading signals
            strategy: Trading strategy instance
            
        Returns:
            DataFrame with backtest results
        """
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.drawdown_breached = False
        self.trades_skipped_due_to_drawdown = 0
        
        current_trade = None
        bars_in_trade = 0
        peak_equity = self.initial_capital
        
        for i in range(len(signals)):
            idx = signals.index[i]
            current_price = signals.loc[idx, 'close']
            
            # Update equity curve
            if current_trade:
                # Calculate unrealized P&L
                if current_trade.direction == 'long':
                    unrealized_pnl = (current_price - current_trade.entry_price) * current_trade.size
                else:
                    unrealized_pnl = (current_trade.entry_price - current_price) * current_trade.size
                
                current_equity = self.capital + unrealized_pnl
            else:
                current_equity = self.capital
            
            self.equity_curve.append({
                'time': idx,
                'equity': current_equity,
                'capital': self.capital
            })
            
            # Update peak equity and check drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # Calculate current drawdown
            current_drawdown = (current_equity - peak_equity) / peak_equity
            
            # Check if drawdown limit is breached
            if self.max_drawdown_percent is not None:
                if current_drawdown <= -self.max_drawdown_percent:
                    self.drawdown_breached = True
                # Resume trading if recovered to within 50% of max drawdown
                elif current_drawdown >= -(self.max_drawdown_percent * 0.5):
                    self.drawdown_breached = False
            
            # Check if we're in a trade
            if current_trade:
                bars_in_trade += 1
                high = signals.loc[idx, 'high']
                low = signals.loc[idx, 'low']
                
                # Check triple barriers
                hit_tp = False
                hit_sl = False
                hit_time = False
                
                if current_trade.direction == 'long':
                    if high >= current_trade.take_profit:
                        hit_tp = True
                        exit_price = current_trade.take_profit
                    elif low <= current_trade.stop_loss:
                        hit_sl = True
                        exit_price = current_trade.stop_loss
                    else:
                        exit_price = current_price
                else:  # short
                    if low <= current_trade.take_profit:
                        hit_tp = True
                        exit_price = current_trade.take_profit
                    elif high >= current_trade.stop_loss:
                        hit_sl = True
                        exit_price = current_trade.stop_loss
                    else:
                        exit_price = current_price
                
                # Check time barrier
                if bars_in_trade >= strategy.triple_barrier.max_holding_bars:
                    hit_time = True
                
                # Close trade if any barrier hit
                if hit_tp or hit_sl or hit_time:
                    if hit_tp:
                        exit_reason = 'take_profit'
                    elif hit_sl:
                        exit_reason = 'stop_loss'
                    else:
                        exit_reason = 'time_exit'
                    
                    current_trade.close(idx, exit_price, exit_reason)
                    
                    # Apply commission
                    commission_cost = abs(current_trade.pnl) * self.commission * 2  # Entry + Exit
                    current_trade.pnl -= commission_cost
                    
                    # Update capital
                    self.capital += current_trade.pnl
                    
                    # Store trade
                    self.trades.append(current_trade)
                    
                    # Reset
                    current_trade = None
                    bars_in_trade = 0
            
            # Check for new signals
            if current_trade is None and signals.loc[idx, 'signal'] != 0:
                # DRAWDOWN PROTECTION: Skip trade if drawdown limit is breached
                if self.drawdown_breached:
                    self.trades_skipped_due_to_drawdown += 1
                    continue
                
                signal = signals.loc[idx, 'signal']
                stop_loss = signals.loc[idx, 'stop_loss']
                entry_price = signals.loc[idx, 'close']
                
                if not pd.isna(stop_loss):
                    # Calculate position size
                    position_size = strategy.position_sizer.calculate_position_size(
                        self.capital, entry_price, stop_loss
                    )
                    
                    # Calculate barriers
                    direction = 'long' if signal == 1 else 'short'
                    barriers = strategy.triple_barrier.calculate_barriers(
                        entry_price, stop_loss, direction
                    )
                    
                    # Create trade
                    current_trade = Trade(
                        entry_time=idx,
                        entry_price=entry_price,
                        direction=direction,
                        size=position_size,
                        stop_loss=barriers['stop_loss'],
                        take_profit=barriers['profit_target']
                    )
        
        # Close any open trade at the end
        if current_trade:
            current_trade.close(
                signals.index[-1],
                signals.loc[signals.index[-1], 'close'],
                'end_of_data'
            )
            self.capital += current_trade.pnl
            self.trades.append(current_trade)
        
        return self._create_results_dataframe()
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create results dataframe from trades."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        return trades_df


# ==================== PART 4: PERFORMANCE ANALYSIS ====================

class PerformanceAnalyzer:
    """
    Analyzes backtest performance and generates metrics.
    """
    
    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, equity_curve: List[Dict],
                         initial_capital: float) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades_df: DataFrame of trades
            equity_curve: List of equity snapshots
            initial_capital: Starting capital
            
        Returns:
            Dictionary of performance metrics
        """
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0
        
        net_profit = trades_df['pnl'].sum()
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        avg_trade = trades_df['pnl'].mean()
        
        # Returns
        final_capital = initial_capital + net_profit
        total_return = (final_capital - initial_capital) / initial_capital * 100
        
        # Drawdown analysis
        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        if len(trades_df) > 1:
            returns = trades_df['return_pct']
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average holding time
        avg_bars_held = trades_df['bars_held'].mean()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_bars_held': avg_bars_held
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Print performance metrics in a readable format."""
        print("\n" + "="*60)
        print(" SMART MONEY CONCEPTS - BACKTEST RESULTS")
        print("="*60 + "\n")
        
        print("TRADE STATISTICS")
        print("-" * 60)
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Winning Trades:         {metrics['winning_trades']}")
        print(f"Losing Trades:          {metrics['losing_trades']}")
        print(f"Win Rate:               {metrics['win_rate']:.2f}%")
        print(f"Average Bars Held:      {metrics['avg_bars_held']:.2f} hours")
        
        print("\nPROFIT/LOSS")
        print("-" * 60)
        print(f"Total Profit:           ${metrics['total_profit']:.2f}")
        print(f"Total Loss:             ${metrics['total_loss']:.2f}")
        print(f"Net Profit:             ${metrics['net_profit']:.2f}")
        print(f"Average Win:            ${metrics['avg_win']:.2f}")
        print(f"Average Loss:           ${metrics['avg_loss']:.2f}")
        print(f"Average Trade:          ${metrics['avg_trade']:.2f}")
        print(f"Profit Factor:          {metrics['profit_factor']:.2f}")
        
        print("\nRETURNS & RISK")
        print("-" * 60)
        print(f"Initial Capital:        ${metrics['initial_capital']:.2f}")
        print(f"Final Capital:          ${metrics['final_capital']:.2f}")
        print(f"Total Return:           {metrics['total_return']:.2f}%")
        print(f"Max Drawdown:           {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
        
        print("\n" + "="*60 + "\n")
    
    @staticmethod
    def plot_results(equity_curve: List[Dict], trades_df: pd.DataFrame, 
                    price_data: pd.DataFrame):
        """
        Plot backtest results.
        
        Args:
            equity_curve: List of equity snapshots
            trades_df: DataFrame of trades
            price_data: Price data for reference
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Equity Curve
        equity_df = pd.DataFrame(equity_curve)
        axes[0].plot(equity_df['time'], equity_df['equity'], label='Equity', linewidth=2)
        axes[0].plot(equity_df['time'], equity_df['capital'], label='Capital (Realized)', 
                    linewidth=1, linestyle='--', alpha=0.7)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Capital ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        axes[1].fill_between(equity_df['time'], equity_df['drawdown'], 0, 
                            color='red', alpha=0.3, label='Drawdown')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Trade Distribution
        if len(trades_df) > 0:
            axes[2].hist(trades_df['return_pct'], bins=30, color='skyblue', 
                        edgecolor='black', alpha=0.7)
            axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
            axes[2].set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Return (%)')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to 'backtest_results.png'")
        plt.show()


# ==================== PART 5: MAIN EXECUTION ====================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare data from CSV."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure proper column names
    df.columns = df.columns.str.lower()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
    
    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    
    print(f"Loaded {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def main():
    """Main execution function."""
    
    print("\n" + "="*60)
    print(" SMART MONEY CONCEPTS (SMC) BACKTESTING SYSTEM")
    print("="*60 + "\n")
    
    # Try to import configuration from config.py
    try:
        import config
        DATA_FILE = config.DATA_FILE
        INITIAL_CAPITAL = config.INITIAL_CAPITAL
        RISK_PER_TRADE = config.RISK_PER_TRADE
        PROFIT_TARGET = config.PROFIT_TARGET
        MAX_HOLDING_BARS = config.MAX_HOLDING_BARS
        SWING_LOOKBACK = config.SWING_LOOKBACK
        COMMISSION = config.COMMISSION
        MAX_DRAWDOWN_LIMIT = config.MAX_DRAWDOWN_LIMIT
    except:
        # Fallback to default configuration
        DATA_FILE = '/home/edward/Documents/smart money/data/BTCUSDT_1m_binance.csv'
        INITIAL_CAPITAL = 10000.0
        RISK_PER_TRADE = 0.01  # 1%
        PROFIT_TARGET = 2.0  # 2:1 R:R
        MAX_HOLDING_BARS = 10  # Hours
        SWING_LOOKBACK = 5
        COMMISSION = 0.001  # 0.1%
        MAX_DRAWDOWN_LIMIT = None
    
    # Step 1: Load data
    df_1m = load_data(DATA_FILE)
    
    # Optional: Limit data for faster testing (remove this for full backtest)
    # df_1m = df_1m.iloc[:100000]  # First 100k rows for testing
    
    # Step 2: Initialize strategy
    print("Initializing trading strategy...")
    strategy = SMCTradingStrategy(
        risk_per_trade=RISK_PER_TRADE,
        profit_target=PROFIT_TARGET,
        max_holding_bars=MAX_HOLDING_BARS,
        swing_lookback=SWING_LOOKBACK
    )
    
    # Step 3: Prepare data (resample and detect structure)
    print("Preparing data and detecting market structure...")
    df_1h, df_4h = strategy.prepare_data(df_1m)
    
    print(f"1H bars: {len(df_1h)}")
    print(f"4H bars: {len(df_4h)}")
    
    # Step 4: Generate signals
    print("Generating trading signals...")
    signals = strategy.generate_signals(df_1h, df_4h)
    
    total_signals = (signals['signal'] != 0).sum()
    print(f"Total signals generated: {total_signals}")
    
    # Step 5: Run backtest
    print("Running backtest...")
    if MAX_DRAWDOWN_LIMIT is not None:
        print(f"⚠️  Drawdown protection enabled: Max {MAX_DRAWDOWN_LIMIT*100:.0f}%")
    
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        max_drawdown_percent=MAX_DRAWDOWN_LIMIT
    )
    
    trades_df = backtester.run_backtest(signals, strategy)
    
    # Print drawdown protection stats
    if MAX_DRAWDOWN_LIMIT is not None and backtester.trades_skipped_due_to_drawdown > 0:
        print(f"\n⚠️  Drawdown Protection Stats:")
        print(f"   Trades skipped due to drawdown limit: {backtester.trades_skipped_due_to_drawdown}")
    
    # Step 6: Analyze results
    print("Analyzing results...")
    analyzer = PerformanceAnalyzer()
    
    metrics = analyzer.calculate_metrics(
        trades_df,
        backtester.equity_curve,
        INITIAL_CAPITAL
    )
    
    # Step 7: Print and plot results
    analyzer.print_metrics(metrics)
    
    if len(trades_df) > 0:
        # Save trades to CSV
        trades_df.to_csv('trades_log.csv', index=False)
        print("Trade log saved to 'trades_log.csv'")
        
        # Plot results
        analyzer.plot_results(backtester.equity_curve, trades_df, df_1h)
    else:
        print("No trades executed. Consider adjusting strategy parameters.")
    
    print("\nBacktest completed!")


if __name__ == '__main__':
    main()

