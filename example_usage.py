"""
ตัวอย่างการใช้งาน Stock Analysis System
Example Usage of Stock Analysis System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stock_analysis_system import (
    StockAnalysisSystem,
    DataPreprocessor,
    FactorModel,
    RiskAnalyzer,
    PortfolioOptimizer,
    ExecutionAnalyzer
)

# กำหนด random seed เพื่อให้ผลลัพธ์ซ้ำได้
np.random.seed(42)


def generate_sample_data(n_stocks=5, n_days=500):
    """
    สร้างข้อมูลตัวอย่างสำหรับทดสอบระบบ
    """
    # สร้างราคาหุ้นจำลอง (Geometric Brownian Motion)
    stock_names = [f'STOCK_{chr(65+i)}' for i in range(n_stocks)]
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

    # พารามิเตอร์สำหรับแต่ละหุ้น
    S0 = np.array([100, 150, 80, 200, 120])  # ราคาเริ่มต้น
    mu = np.array([0.10, 0.15, 0.08, 0.12, 0.11])  # Expected return (annualized)
    sigma = np.array([0.20, 0.30, 0.15, 0.25, 0.22])  # Volatility (annualized)

    prices = {}
    dt = 1/252  # Daily

    for i, stock in enumerate(stock_names):
        price_path = [S0[i]]
        for _ in range(n_days - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = mu[i] * price_path[-1] * dt + sigma[i] * price_path[-1] * dW
            price_path.append(price_path[-1] + dS)
        prices[stock] = price_path

    prices_df = pd.DataFrame(prices, index=dates)

    # สร้างข้อมูล Market Cap และ Book-to-Market
    market_caps = pd.Series({
        'STOCK_A': 10e9,
        'STOCK_B': 5e9,
        'STOCK_C': 50e9,
        'STOCK_D': 2e9,
        'STOCK_E': 20e9
    })

    book_to_market = pd.Series({
        'STOCK_A': 0.8,
        'STOCK_B': 1.2,
        'STOCK_C': 0.5,
        'STOCK_D': 1.5,
        'STOCK_E': 0.9
    })

    return prices_df, market_caps, book_to_market


def example_1_basic_analysis():
    """
    ตัวอย่างที่ 1: การวิเคราะห์พื้นฐาน
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 1: การวิเคราะห์หุ้นแบบครบวงจร")
    print("="*80)

    # สร้างข้อมูลตัวอย่าง
    prices, market_caps, book_to_market = generate_sample_data()

    # สร้างระบบวิเคราะห์
    system = StockAnalysisSystem()

    # วิเคราะห์
    results = system.analyze_stock(prices, market_caps, book_to_market)

    # แสดงผลลัพธ์
    print("\n📊 สรุปผลการวิเคราะห์:")
    print("-" * 80)

    # Portfolio Optimization Results
    mv_result = results['portfolio_optimization']['mean_variance']
    print("\n🎯 Mean-Variance Optimal Portfolio:")
    print(f"  Expected Return: {mv_result['expected_return']*100:.2f}%")
    print(f"  Volatility: {mv_result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {mv_result['sharpe_ratio']:.3f}")
    print(f"\n  Optimal Weights:")
    for i, weight in enumerate(mv_result['weights']):
        print(f"    STOCK_{chr(65+i)}: {weight*100:.2f}%")

    # Risk Parity Weights
    rp_weights = results['portfolio_optimization']['risk_parity_weights']
    print(f"\n⚖️  Risk Parity Weights:")
    for i, weight in enumerate(rp_weights):
        print(f"    STOCK_{chr(65+i)}: {weight*100:.2f}%")

    # VaR Results
    print(f"\n⚠️  Value at Risk (99% confidence):")
    for stock, var_result in results['risk_analysis']['var'].items():
        print(f"    {stock}: {var_result['VaR']*100:.2f}% (Daily)")

    # Factor Analysis
    if results['factor_analysis']:
        print(f"\n📈 Factor Analysis (Alpha):")
        for stock, factor_result in results['factor_analysis'].items():
            alpha = factor_result['alpha'] * 252 * 100  # Annualized
            print(f"    {stock}: {alpha:.2f}% (annualized)")

    return results


def example_2_data_preprocessing():
    """
    ตัวอย่างที่ 2: การประมวลผลข้อมูล
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 2: การประมวลผลข้อมูลเบื้องต้น")
    print("="*80)

    prices, _, _ = generate_sample_data()
    preprocessor = DataPreprocessor()

    # 1. แปลงเป็น Log Returns
    log_returns = preprocessor.to_log_returns(prices)
    print(f"\n✓ แปลงเป็น Log Returns สำเร็จ")
    print(f"  Shape: {log_returns.shape}")
    print(f"  Mean Daily Return: {log_returns.mean().mean()*100:.4f}%")

    # 2. ตรวจจับ Outliers
    outlier_mask = preprocessor.detect_outliers(log_returns, threshold=3.0)

    # 3. จัดการข้อมูลที่ขาดหาย
    log_returns_clean = preprocessor.handle_missing_data(log_returns)

    # 4. QR Decomposition
    Q, R = preprocessor.qr_decomposition(log_returns.values)
    print(f"\n✓ QR Decomposition สำเร็จ")
    print(f"  Q shape: {Q.shape}")
    print(f"  R shape: {R.shape}")

    return log_returns


def example_3_factor_model():
    """
    ตัวอย่างที่ 3: Factor Modeling และ Alpha
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 3: Factor Modeling และการหา Alpha")
    print("="*80)

    prices, market_caps, book_to_market = generate_sample_data()
    preprocessor = DataPreprocessor()
    factor_model = FactorModel()

    log_returns = preprocessor.to_log_returns(prices)

    # สร้าง Fama-French Factors
    factors = factor_model.create_fama_french_factors(
        log_returns, market_caps, book_to_market
    )

    print(f"\n✓ สร้าง Fama-French Factors สำเร็จ")
    print(f"\nFactor Statistics:")
    print(factors.describe())

    # วิเคราะห์การถดถอยสำหรับหุ้นตัวแรก
    stock = 'STOCK_A'
    regression_result = factor_model.regression_analysis(
        log_returns[stock], factors
    )

    print(f"\n📊 Regression Results for {stock}:")
    print(f"  Alpha: {regression_result['alpha']*252*100:.2f}% (annualized)")
    print(f"  R-squared: {regression_result['r_squared']:.4f}")
    print(f"\n  Factor Loadings (Betas):")
    for i, factor_name in enumerate(regression_result['factor_names']):
        beta = regression_result['betas'][i]
        t_stat = regression_result['t_stats'][i+1]
        p_val = regression_result['p_values'][i+1]
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        print(f"    {factor_name}: {beta:.4f} (t={t_stat:.2f}) {sig}")

    return regression_result


def example_4_risk_analysis():
    """
    ตัวอย่างที่ 4: การวิเคราะห์ความเสี่ยง
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 4: การวิเคราะห์ความเสี่ยงและความผันผวน")
    print("="*80)

    prices, _, _ = generate_sample_data()
    preprocessor = DataPreprocessor()
    risk_analyzer = RiskAnalyzer()

    log_returns = preprocessor.to_log_returns(prices)

    # 1. GARCH(1,1) สำหรับหุ้นตัวแรก
    stock = 'STOCK_A'
    garch_result = risk_analyzer.estimate_garch_11(log_returns[stock])

    print(f"\n📈 GARCH(1,1) Results for {stock}:")
    print(f"  ω (omega): {garch_result['omega']:.6f}")
    print(f"  α (alpha): {garch_result['alpha']:.4f}")
    print(f"  β (beta): {garch_result['beta']:.4f}")
    print(f"  Persistence (α+β): {garch_result['persistence']:.4f}")
    if garch_result['unconditional_variance']:
        print(f"  Unconditional Variance: {garch_result['unconditional_variance']:.6f}")

    # 2. Value at Risk (VaR)
    var_methods = ['parametric', 'historical', 'monte_carlo']
    print(f"\n⚠️  Value at Risk (99% confidence) for {stock}:")

    for method in var_methods:
        var_result = risk_analyzer.calculate_var(
            log_returns[stock], confidence_level=0.99, method=method
        )
        print(f"  {method.capitalize():15s}: VaR = {var_result['VaR']*100:.2f}%, " +
              f"ES = {var_result['Expected_Shortfall']*100:.2f}%")

    # 3. Correlation Matrix Analysis
    corr_analysis = risk_analyzer.correlation_matrix_analysis(log_returns)

    print(f"\n🔗 Correlation Matrix:")
    print(corr_analysis['correlation_matrix'])

    print(f"\n🔍 Principal Components (Eigenvalues):")
    for i, (eval, var_ratio) in enumerate(zip(
        corr_analysis['eigenvalues'][:3],
        corr_analysis['explained_variance_ratio'][:3]
    )):
        print(f"  PC{i+1}: {eval:.4f} (explains {var_ratio*100:.1f}% of variance)")

    return corr_analysis


def example_5_portfolio_optimization():
    """
    ตัวอย่างที่ 5: การหาพอร์ตโฟลิโอที่เหมาะสม
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 5: Portfolio Optimization")
    print("="*80)

    prices, _, _ = generate_sample_data()
    preprocessor = DataPreprocessor()
    optimizer = PortfolioOptimizer()

    log_returns = preprocessor.to_log_returns(prices)
    expected_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    # 1. Mean-Variance Optimization
    mv_result = optimizer.mean_variance_optimization(
        expected_returns.values, cov_matrix.values
    )

    print(f"\n🎯 Mean-Variance Optimal Portfolio:")
    print(f"  Expected Return: {mv_result['expected_return']*100:.2f}%")
    print(f"  Volatility: {mv_result['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {mv_result['sharpe_ratio']:.3f}")

    # 2. Efficient Frontier
    efficient_frontier = optimizer.efficient_frontier(
        expected_returns.values, cov_matrix.values, n_points=20
    )

    print(f"\n📈 Efficient Frontier:")
    print(f"  Min Return: {efficient_frontier['return'].min()*100:.2f}%")
    print(f"  Max Return: {efficient_frontier['return'].max()*100:.2f}%")
    print(f"  Min Volatility: {efficient_frontier['volatility'].min()*100:.2f}%")

    # 3. Kelly Criterion
    win_prob = 0.55
    win_return = 0.02
    loss_return = -0.015

    kelly_fraction = optimizer.kelly_criterion(win_prob, win_return, loss_return)
    print(f"\n💰 Kelly Criterion:")
    print(f"  Win Probability: {win_prob*100:.0f}%")
    print(f"  Win/Loss Ratio: {abs(win_return/loss_return):.2f}")
    print(f"  Optimal Position Size: {kelly_fraction*100:.1f}% of capital")

    # 4. Risk Parity
    rp_weights = optimizer.risk_parity(cov_matrix.values)

    print(f"\n⚖️  Risk Parity Portfolio:")
    for i, (stock, weight) in enumerate(zip(log_returns.columns, rp_weights)):
        print(f"  {stock}: {weight*100:.2f}%")

    return efficient_frontier


def example_6_execution_analysis():
    """
    ตัวอย่างที่ 6: Execution และ Monte Carlo Simulation
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 6: Execution Analysis และ Monte Carlo")
    print("="*80)

    prices, _, _ = generate_sample_data()
    preprocessor = DataPreprocessor()
    executor = ExecutionAnalyzer()

    log_returns = preprocessor.to_log_returns(prices)

    # 1. Monte Carlo Simulation
    stock = 'STOCK_A'
    mu = log_returns[stock].mean() * 252
    sigma = log_returns[stock].std() * np.sqrt(252)
    S0 = prices[stock].iloc[-1]

    mc_result = executor.monte_carlo_simulation(
        mu, sigma, S0, T=1.0, n_simulations=10000
    )

    print(f"\n🎲 Monte Carlo Simulation for {stock}:")
    print(f"  Current Price: ${S0:.2f}")
    print(f"  Expected Price (1 year): ${mc_result['mean_final_price']:.2f}")
    print(f"  Median Price: ${mc_result['median_final_price']:.2f}")
    print(f"  5th Percentile: ${mc_result['percentile_5']:.2f}")
    print(f"  95th Percentile: ${mc_result['percentile_95']:.2f}")

    # 2. Market Impact
    order_size = 100000  # shares
    avg_daily_volume = 1000000  # shares
    volatility = sigma

    impact = executor.calculate_market_impact(
        order_size, avg_daily_volume, volatility
    )

    print(f"\n💸 Market Impact Analysis:")
    print(f"  Order Size: {order_size:,.0f} shares")
    print(f"  Participation Rate: {impact['participation_rate']*100:.1f}%")
    print(f"  Temporary Impact: {impact['temporary_impact']*100:.3f}%")
    print(f"  Permanent Impact: {impact['permanent_impact']*100:.3f}%")
    print(f"  Total Impact: {impact['total_impact']*100:.3f}%")
    print(f"  Estimated Slippage: {impact['estimated_slippage_bps']:.1f} bps")

    # 3. Backtest Simple Strategy
    # สร้าง signal ง่ายๆ: Long when return > median, Short when < median
    median_return = log_returns[stock].median()
    signals = pd.Series(0, index=log_returns.index)
    signals[log_returns[stock] > median_return] = 1
    signals[log_returns[stock] < median_return] = -1

    backtest_result = executor.backtest_strategy(
        signals, log_returns[stock], transaction_cost=0.001
    )

    print(f"\n📊 Backtest Results (Simple Mean-Reversion Strategy):")
    print(f"  Total Return: {backtest_result['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {backtest_result['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {backtest_result['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {backtest_result['win_rate']*100:.1f}%")

    return mc_result


def example_7_visualization():
    """
    ตัวอย่างที่ 7: การแสดงผลด้วยกราฟ
    """
    print("\n" + "="*80)
    print("ตัวอย่างที่ 7: Visualization")
    print("="*80)

    prices, _, _ = generate_sample_data()
    preprocessor = DataPreprocessor()
    optimizer = PortfolioOptimizer()

    log_returns = preprocessor.to_log_returns(prices)
    expected_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    # Create efficient frontier
    efficient_frontier = optimizer.efficient_frontier(
        expected_returns.values, cov_matrix.values, n_points=50
    )

    # Plot
    plt.figure(figsize=(12, 8))

    # Subplot 1: Price Evolution
    plt.subplot(2, 2, 1)
    normalized_prices = prices / prices.iloc[0] * 100
    for col in normalized_prices.columns:
        plt.plot(normalized_prices.index, normalized_prices[col], label=col, alpha=0.7)
    plt.title('Stock Price Evolution (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Price (Base = 100)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Efficient Frontier
    plt.subplot(2, 2, 2)
    plt.scatter(efficient_frontier['volatility']*100,
               efficient_frontier['return']*100,
               c=efficient_frontier['sharpe_ratio'],
               cmap='viridis', s=50)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Expected Return (%)')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Correlation Matrix
    plt.subplot(2, 2, 3)
    corr = log_returns.corr()
    im = plt.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Matrix')

    # Subplot 4: Return Distribution
    plt.subplot(2, 2, 4)
    for col in log_returns.columns:
        plt.hist(log_returns[col]*100, bins=50, alpha=0.5, label=col)
    plt.title('Return Distribution')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('stock_analysis_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ บันทึกกราฟที่ 'stock_analysis_visualization.png'")
    plt.close()


def main():
    """
    รันตัวอย่างทั้งหมด
    """
    print("\n" + "="*80)
    print("🚀 Stock Analysis System - ระบบวิเคราะห์หุ้นแบบครบวงจร")
    print("   Based on MIT Financial Engineering Principles")
    print("="*80)

    # รันตัวอย่างทั้งหมด
    example_1_basic_analysis()
    example_2_data_preprocessing()
    example_3_factor_model()
    example_4_risk_analysis()
    example_5_portfolio_optimization()
    example_6_execution_analysis()

    # Visualization
    try:
        example_7_visualization()
    except Exception as e:
        print(f"\n⚠️  ไม่สามารถสร้างกราฟได้: {e}")

    print("\n" + "="*80)
    print("✅ การวิเคราะห์ทั้งหมดเสร็จสมบูรณ์!")
    print("="*80)
    print("\n📚 คุณสมบัติของระบบ:")
    print("  ✓ Data Preprocessing (Log Returns, Outlier Detection, QR Decomposition)")
    print("  ✓ Factor Modeling (Fama-French, Regression Analysis)")
    print("  ✓ Risk Analysis (GARCH, VaR, Correlation Matrix)")
    print("  ✓ Portfolio Optimization (Mean-Variance, Risk Parity, Kelly Criterion)")
    print("  ✓ Execution Analysis (Monte Carlo, Market Impact, Backtesting)")
    print("\n")


if __name__ == "__main__":
    main()
