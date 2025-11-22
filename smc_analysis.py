"""
SMC Trading System - Advanced Analysis & Visualization
========================================================

Enhanced analysis tools for the Smart Money Concepts trading system.
Includes monthly breakdowns, win/loss analysis, and advanced visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class SMCAnalyzer:
    """Advanced analysis for SMC backtest results."""
    
    def __init__(self, trades_file: str = 'trades_log.csv'):
        """Load trades from CSV."""
        self.trades = pd.read_csv(trades_file)
        self.trades['entry_time'] = pd.to_datetime(self.trades['entry_time'])
        self.trades['exit_time'] = pd.to_datetime(self.trades['exit_time'])
        self.trades['year'] = self.trades['entry_time'].dt.year
        self.trades['month'] = self.trades['entry_time'].dt.month
        self.trades['year_month'] = self.trades['entry_time'].dt.to_period('M')
    
    def monthly_performance(self) -> pd.DataFrame:
        """Calculate monthly performance metrics."""
        monthly = self.trades.groupby('year_month').agg({
            'pnl': ['sum', 'mean', 'count'],
            'return_pct': 'mean',
            'exit_reason': lambda x: (x == 'take_profit').sum()
        }).round(2)
        
        monthly.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Avg Return %', 'Wins']
        monthly['Win Rate %'] = (monthly['Wins'] / monthly['Trades'] * 100).round(2)
        
        return monthly
    
    def exit_reason_analysis(self):
        """Analyze exit reasons."""
        exit_counts = self.trades['exit_reason'].value_counts()
        exit_pnl = self.trades.groupby('exit_reason')['pnl'].agg(['sum', 'mean', 'count'])
        
        return exit_counts, exit_pnl
    
    def direction_analysis(self):
        """Analyze long vs short performance."""
        direction_perf = self.trades.groupby('direction').agg({
            'pnl': ['sum', 'mean', 'count'],
            'return_pct': 'mean'
        }).round(2)
        
        # Win rates by direction
        for direction in ['long', 'short']:
            mask = self.trades['direction'] == direction
            wins = (self.trades[mask]['pnl'] > 0).sum()
            total = mask.sum()
            print(f"\n{direction.upper()} Performance:")
            print(f"  Total Trades: {total}")
            print(f"  Win Rate: {(wins/total*100):.2f}%")
            print(f"  Total P&L: ${self.trades[mask]['pnl'].sum():.2f}")
            print(f"  Avg P&L: ${self.trades[mask]['pnl'].mean():.2f}")
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive analysis plots."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative P&L over time
        ax1 = fig.add_subplot(gs[0, :])
        self.trades['cumulative_pnl'] = self.trades['pnl'].cumsum()
        ax1.plot(self.trades['exit_time'], self.trades['cumulative_pnl'], 
                linewidth=2, color='darkblue')
        ax1.set_title('Cumulative P&L Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative P&L ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 2. Monthly P&L
        ax2 = fig.add_subplot(gs[1, :])
        monthly = self.trades.groupby('year_month')['pnl'].sum()
        colors = ['green' if x > 0 else 'red' for x in monthly.values]
        ax2.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax2.set_title('Monthly P&L', fontsize=14, fontweight='bold')
        ax2.set_ylabel('P&L ($)')
        ax2.set_xticks(range(len(monthly)))
        ax2.set_xticklabels(monthly.index.astype(str), rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Win/Loss Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        wins = self.trades[self.trades['pnl'] > 0]['pnl']
        losses = self.trades[self.trades['pnl'] < 0]['pnl']
        ax3.hist([wins, losses], bins=30, label=['Wins', 'Losses'], 
                color=['green', 'red'], alpha=0.6)
        ax3.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Exit Reason Distribution
        ax4 = fig.add_subplot(gs[2, 1])
        exit_counts = self.trades['exit_reason'].value_counts()
        colors_exit = {'take_profit': 'green', 'stop_loss': 'red', 
                      'time_exit': 'orange', 'end_of_data': 'gray'}
        colors_list = [colors_exit.get(x, 'blue') for x in exit_counts.index]
        ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
               colors=colors_list, startangle=90)
        ax4.set_title('Exit Reasons', fontsize=12, fontweight='bold')
        
        # 5. Long vs Short Performance
        ax5 = fig.add_subplot(gs[2, 2])
        direction_pnl = self.trades.groupby('direction')['pnl'].sum()
        colors_dir = ['#2ecc71' if x > 0 else '#e74c3c' for x in direction_pnl.values]
        ax5.bar(direction_pnl.index, direction_pnl.values, color=colors_dir, alpha=0.7)
        ax5.set_title('Long vs Short Total P&L', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Total P&L ($)')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Return % Distribution by Exit Reason
        ax6 = fig.add_subplot(gs[3, 0])
        self.trades.boxplot(column='return_pct', by='exit_reason', ax=ax6)
        ax6.set_title('Return % by Exit Reason', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Exit Reason')
        ax6.set_ylabel('Return (%)')
        plt.sca(ax6)
        plt.xticks(rotation=45, ha='right')
        
        # 7. Bars Held Distribution
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.hist(self.trades['bars_held'], bins=20, color='skyblue', 
                edgecolor='black', alpha=0.7)
        ax7.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Hours Held')
        ax7.set_ylabel('Frequency')
        ax7.axvline(x=self.trades['bars_held'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {self.trades['bars_held'].mean():.1f}h")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Win Rate Over Time (rolling 100 trades)
        ax8 = fig.add_subplot(gs[3, 2])
        self.trades['is_win'] = (self.trades['pnl'] > 0).astype(int)
        rolling_wr = self.trades['is_win'].rolling(window=100).mean() * 100
        ax8.plot(range(len(rolling_wr)), rolling_wr, linewidth=2, color='purple')
        ax8.set_title('Rolling Win Rate (100 trades)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Trade Number')
        ax8.set_ylabel('Win Rate (%)')
        ax8.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax8.axhline(y=rolling_wr.mean(), color='red', linestyle='--', 
                   alpha=0.7, label=f'Mean: {rolling_wr.mean():.1f}%')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Smart Money Concepts - Comprehensive Trade Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('smc_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\nComprehensive analysis saved to 'smc_comprehensive_analysis.png'")
        plt.show()
    
    def print_detailed_stats(self):
        """Print detailed statistics."""
        print("\n" + "="*70)
        print(" DETAILED STATISTICS")
        print("="*70 + "\n")
        
        # Overall stats
        print("OVERALL PERFORMANCE")
        print("-" * 70)
        print(f"Period: {self.trades['entry_time'].min()} to {self.trades['exit_time'].max()}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Total Days: {(self.trades['exit_time'].max() - self.trades['entry_time'].min()).days}")
        print(f"Trades per Day: {len(self.trades) / (self.trades['exit_time'].max() - self.trades['entry_time'].min()).days:.2f}")
        
        # Exit reasons
        print("\n\nEXIT REASON BREAKDOWN")
        print("-" * 70)
        exit_counts, exit_pnl = self.exit_reason_analysis()
        for reason in exit_counts.index:
            count = exit_counts[reason]
            pct = count / len(self.trades) * 100
            avg_pnl = exit_pnl.loc[reason, 'mean']
            total_pnl = exit_pnl.loc[reason, 'sum']
            print(f"\n{reason.upper()}:")
            print(f"  Count: {count} ({pct:.1f}%)")
            print(f"  Avg P&L: ${avg_pnl:.2f}")
            print(f"  Total P&L: ${total_pnl:.2f}")
        
        # Best and worst trades
        print("\n\nBEST & WORST TRADES")
        print("-" * 70)
        best = self.trades.nlargest(5, 'pnl')[['entry_time', 'direction', 'entry_price', 
                                                 'exit_price', 'pnl', 'return_pct', 'exit_reason']]
        worst = self.trades.nsmallest(5, 'pnl')[['entry_time', 'direction', 'entry_price', 
                                                   'exit_price', 'pnl', 'return_pct', 'exit_reason']]
        
        print("\nTop 5 Best Trades:")
        print(best.to_string(index=False))
        
        print("\n\nTop 5 Worst Trades:")
        print(worst.to_string(index=False))
        
        # Monthly summary
        print("\n\nMONTHLY PERFORMANCE SUMMARY")
        print("-" * 70)
        monthly = self.monthly_performance()
        print(monthly.tail(12).to_string())  # Last 12 months
        
        print("\n" + "="*70 + "\n")


def main():
    """Run advanced analysis."""
    print("\n" + "="*70)
    print(" SMC TRADING SYSTEM - ADVANCED ANALYSIS")
    print("="*70)
    
    try:
        analyzer = SMCAnalyzer('trades_log.csv')
        
        # Print detailed statistics
        analyzer.print_detailed_stats()
        
        # Direction analysis
        print("\n" + "="*70)
        print(" LONG vs SHORT ANALYSIS")
        print("="*70)
        analyzer.direction_analysis()
        
        # Create comprehensive plots
        print("\n\nGenerating comprehensive analysis plots...")
        analyzer.plot_comprehensive_analysis()
        
        print("\n✓ Analysis complete!")
        
    except FileNotFoundError:
        print("\n❌ Error: trades_log.csv not found. Please run smart_money.py first.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")


if __name__ == '__main__':
    main()

