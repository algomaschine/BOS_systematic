"""
Drawdown Optimizer - Testing Different Drawdown Limits
=======================================================

This tool tests the SMC trading strategy with different maximum drawdown limits
to find the optimal balance between risk protection and profitability.

Usage:
    python drawdown_optimizer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smart_money import (
    SMCTradingStrategy, Backtester, PerformanceAnalyzer, load_data
)
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def test_drawdown_limits(df_1m: pd.DataFrame, 
                        drawdown_limits: list,
                        strategy_params: dict) -> pd.DataFrame:
    """
    Test strategy with different drawdown limits.
    
    Args:
        df_1m: 1-minute OHLC data
        drawdown_limits: List of drawdown limits to test (e.g., [0.10, 0.15, 0.20])
        strategy_params: Dictionary of strategy parameters
        
    Returns:
        DataFrame with results for each drawdown limit
    """
    results = []
    
    print("\n" + "="*80)
    print(" DRAWDOWN LIMIT OPTIMIZATION")
    print("="*80 + "\n")
    
    print(f"Testing {len(drawdown_limits)} different drawdown limits...")
    print(f"Data range: {df_1m['Date'].min()} to {df_1m['Date'].max()}")
    print(f"Total bars: {len(df_1m):,}\n")
    
    # Initialize strategy once
    strategy = SMCTradingStrategy(
        risk_per_trade=strategy_params['risk_per_trade'],
        profit_target=strategy_params['profit_target'],
        max_holding_bars=strategy_params['max_holding_bars'],
        swing_lookback=strategy_params['swing_lookback']
    )
    
    # Prepare data once (expensive operation)
    print("Preparing data and detecting market structure...")
    df_1h, df_4h = strategy.prepare_data(df_1m)
    
    print("Generating trading signals...")
    signals = strategy.generate_signals(df_1h, df_4h)
    
    total_signals = (signals['signal'] != 0).sum()
    print(f"Total signals generated: {total_signals}\n")
    
    # Test each drawdown limit
    for i, dd_limit in enumerate(drawdown_limits, 1):
        if dd_limit is None:
            print(f"\n[{i}/{len(drawdown_limits)}] Testing: No Limit (Baseline)")
        else:
            print(f"\n[{i}/{len(drawdown_limits)}] Testing Drawdown Limit: {dd_limit*100:.0f}%")
        print("-" * 80)
        
        # Run backtest with this drawdown limit
        backtester = Backtester(
            initial_capital=strategy_params['initial_capital'],
            commission=strategy_params['commission'],
            max_drawdown_percent=dd_limit
        )
        
        trades_df = backtester.run_backtest(signals, strategy)
        
        if len(trades_df) == 0:
            print("⚠️  No trades executed!")
            continue
        
        # Calculate metrics
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(
            trades_df,
            backtester.equity_curve,
            strategy_params['initial_capital']
        )
        
        # Store results
        result = {
            'drawdown_limit': dd_limit,
            'drawdown_limit_pct': dd_limit * 100 if dd_limit else 999,
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'net_profit': metrics['net_profit'],
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'avg_trade': metrics['avg_trade'],
            'trades_skipped': backtester.trades_skipped_due_to_drawdown
        }
        
        results.append(result)
        
        # Print summary
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Trades Skipped: {backtester.trades_skipped_due_to_drawdown}")
        print(f"   Win Rate: {metrics['win_rate']:.2f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Net Profit: ${metrics['net_profit']:,.2f}")
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    results_df = pd.DataFrame(results)
    
    return results_df


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print(" DRAWDOWN LIMIT OPTIMIZER")
    print("="*80)
    
    # Load configuration
    DATA_FILE = config.DATA_FILE
    INITIAL_CAPITAL = config.INITIAL_CAPITAL
    RISK_PER_TRADE = config.RISK_PER_TRADE
    PROFIT_TARGET = config.PROFIT_TARGET
    MAX_HOLDING_BARS = config.MAX_HOLDING_BARS
    SWING_LOOKBACK = config.SWING_LOOKBACK
    COMMISSION = config.COMMISSION
    
    # Define drawdown limits to test
    DRAWDOWN_LIMITS = [
        None,  # No limit (baseline)
        0.50,  # 50%
        0.40,  # 40%
        0.30,  # 30%
        0.20,  # 20%
        0.15,  # 15%
        0.10,  # 10%
    ]
    
    print(f"\nConfiguration:")
    print(f"  Data File: {DATA_FILE}")
    print(f"  Drawdown Limits to Test: {len(DRAWDOWN_LIMITS)}")
    
    # Load data
    df_1m = load_data(DATA_FILE)
    
    # Prepare strategy parameters
    strategy_params = {
        'initial_capital': INITIAL_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'profit_target': PROFIT_TARGET,
        'max_holding_bars': MAX_HOLDING_BARS,
        'swing_lookback': SWING_LOOKBACK,
        'commission': COMMISSION
    }
    
    # Test different drawdown limits
    results_df = test_drawdown_limits(df_1m, DRAWDOWN_LIMITS, strategy_params)
    
    # Save results
    results_df.to_csv('drawdown_optimization_results.csv', index=False)
    print("\n✓ Results saved to 'drawdown_optimization_results.csv'")
    
    print("\n" + "="*80)
    print(" OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
