"""
SMC Trading System - Configuration File
========================================

Adjust these parameters to customize the trading strategy.
"""

# ==================== DATA CONFIGURATION ====================

# Path to your 1-minute OHLCV data
DATA_FILE = '/home/edward/Documents/smart money/data/BTCUSDT_1m_binance.csv'

# Date range for backtesting (None = use all available data)
START_DATE = None  # Example: '2020-01-01'
END_DATE = None    # Example: '2024-12-31'

# For faster testing, limit the number of rows (None = use all data)
MAX_ROWS = None  # Example: 100000 for testing


# ==================== RISK MANAGEMENT ====================

# Initial capital for backtesting
INITIAL_CAPITAL = 10000.0

# Risk per trade as a percentage of capital
# 0.01 = 1%, 0.02 = 2%, etc.
# Recommended: 0.01 to 0.02 (never exceed 0.05)
RISK_PER_TRADE = 0.01

# Trading commission/fees (as decimal)
# 0.001 = 0.1%, 0.0004 = 0.04% (typical for Binance VIP)
COMMISSION = 0.001

# Maximum risk per trade as percentage of entry price
# This prevents taking trades with stops too far away
MAX_RISK_PERCENT = 0.03  # 3%


# ==================== STRATEGY PARAMETERS ====================

# Swing Detection
# Number of bars to look left and right for swing highs/lows
# Higher = more significant swings, fewer signals
# Lower = more swings detected, more signals
SWING_LOOKBACK = 5  # Recommended: 3-7

# Fair Value Gap (FVG) Detection
# Minimum gap size as percentage of price to be considered valid FVG
# 0.001 = 0.1%, 0.002 = 0.2%
MIN_FVG_SIZE = 0.001

# Triple Barrier Method
# Profit target as multiple of risk (R:R ratio)
# 2.0 = 2:1, 3.0 = 3:1
PROFIT_TARGET = 2.0  # Recommended: 1.5 to 3.0

# Maximum holding period (in hours for 1H timeframe)
# After this many bars, position closes at market regardless of P&L
MAX_HOLDING_BARS = 10  # Recommended: 5-15


# ==================== ENTRY FILTERS ====================

# Require recent BOS confirmation on 1H
# How many bars to look back for 1H BOS confirmation
BOS_CONFIRMATION_LOOKBACK = 5  # bars

# Require swing low/high proximity for entries
# How many bars to look back for swing points
SWING_PROXIMITY_LOOKBACK = 20  # bars


# ==================== ADVANCED SETTINGS ====================

# Position Sizing Method
# 'fixed_fractional' = risk fixed % per trade
# 'kelly' = Kelly Criterion (more aggressive)
POSITION_SIZING_METHOD = 'fixed_fractional'

# Enable/disable long trades (True/False)
ENABLE_LONG_TRADES = True

# Enable/disable short trades (True/False)
ENABLE_SHORT_TRADES = True


# ==================== OUTPUT SETTINGS ====================

# Save detailed trade log
SAVE_TRADE_LOG = True
TRADE_LOG_FILE = 'trades_log.csv'

# Generate plots
GENERATE_PLOTS = True

# Save equity curve data
SAVE_EQUITY_CURVE = True
EQUITY_CURVE_FILE = 'equity_curve.csv'


# ==================== DISPLAY SETTINGS ====================

# Print progress during backtest
VERBOSE = True

# Show plots after backtest completes
SHOW_PLOTS = True


# ==================== OPTIMIZATION PARAMETERS ====================

# For parameter optimization (advanced users)
OPTIMIZATION_PARAMS = {
    'swing_lookback': [3, 5, 7],
    'profit_target': [1.5, 2.0, 2.5, 3.0],
    'max_holding_bars': [5, 10, 15],
    'risk_per_trade': [0.01, 0.015, 0.02]
}


# ==================== VALIDATION ====================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.1:
        errors.append("RISK_PER_TRADE must be between 0 and 0.1 (0% and 10%)")
    
    if PROFIT_TARGET <= 1.0:
        errors.append("PROFIT_TARGET must be greater than 1.0")
    
    if MAX_HOLDING_BARS <= 0:
        errors.append("MAX_HOLDING_BARS must be positive")
    
    if SWING_LOOKBACK < 2:
        errors.append("SWING_LOOKBACK must be at least 2")
    
    if not ENABLE_LONG_TRADES and not ENABLE_SHORT_TRADES:
        errors.append("At least one of ENABLE_LONG_TRADES or ENABLE_SHORT_TRADES must be True")
    
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == '__main__':
    """Test configuration validation."""
    print("\nValidating configuration...")
    if validate_config():
        print("✓ Configuration is valid!")
        print(f"\nCurrent Settings:")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"  Risk per Trade: {RISK_PER_TRADE*100:.1f}%")
        print(f"  Profit Target: {PROFIT_TARGET}:1")
        print(f"  Max Holding: {MAX_HOLDING_BARS} hours")
        print(f"  Swing Lookback: {SWING_LOOKBACK} bars")
    else:
        print("\n❌ Please fix configuration errors before running backtest.")

