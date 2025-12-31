"""
Simple Test Suite for Stock Analysis System
ชุดทดสอบง่ายๆ เพื่อตรวจสอบว่าระบบทำงานถูกต้อง
"""

import numpy as np
import pandas as pd
from stock_analysis_system import (
    DataPreprocessor,
    FactorModel,
    RiskAnalyzer,
    PortfolioOptimizer,
    ExecutionAnalyzer,
    StockAnalysisSystem
)

# Set random seed for reproducibility
np.random.seed(42)


def test_data_preprocessor():
    """Test Data Preprocessing module"""
    print("\n[TEST 1] Testing DataPreprocessor...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'A': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'B': 150 + np.cumsum(np.random.randn(100) * 0.7)
    }, index=dates)

    preprocessor = DataPreprocessor()

    # Test log returns
    log_returns = preprocessor.to_log_returns(prices)
    assert log_returns.shape[0] == 99, "Log returns shape incorrect"
    assert not log_returns.isnull().all().any(), "Log returns contain all NaN"

    # Test outlier detection
    outlier_mask = preprocessor.detect_outliers(log_returns)
    assert outlier_mask.shape == log_returns.shape, "Outlier mask shape incorrect"

    # Test QR decomposition
    Q, R = preprocessor.qr_decomposition(log_returns.values)
    assert Q.shape[1] == R.shape[0], "QR decomposition dimensions mismatch"

    # Test full pipeline
    result = preprocessor.preprocess_pipeline(prices)
    assert 'log_returns' in result, "Pipeline missing log_returns"
    assert 'Q' in result, "Pipeline missing Q matrix"
    assert 'R' in result, "Pipeline missing R matrix"

    print("  ✓ All DataPreprocessor tests passed")
    return True


def test_factor_model():
    """Test Factor Model module"""
    print("\n[TEST 2] Testing FactorModel...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.01,
        index=dates,
        columns=['A', 'B', 'C', 'D', 'E']
    )

    market_caps = pd.Series([10e9, 5e9, 50e9, 2e9, 20e9], index=returns.columns)
    book_to_market = pd.Series([0.8, 1.2, 0.5, 1.5, 0.9], index=returns.columns)

    factor_model = FactorModel()

    # Test Fama-French factors
    factors = factor_model.create_fama_french_factors(returns, market_caps, book_to_market)
    assert 'MKT' in factors.columns, "Missing MKT factor"
    assert 'SMB' in factors.columns, "Missing SMB factor"
    assert 'HML' in factors.columns, "Missing HML factor"

    # Test regression analysis
    regression_result = factor_model.regression_analysis(returns['A'], factors)
    assert 'alpha' in regression_result, "Regression missing alpha"
    assert 'betas' in regression_result, "Regression missing betas"
    assert 'r_squared' in regression_result, "Regression missing r_squared"
    assert 0 <= regression_result['r_squared'] <= 1, "R-squared out of valid range"

    print("  ✓ All FactorModel tests passed")
    return True


def test_risk_analyzer():
    """Test Risk Analyzer module"""
    print("\n[TEST 3] Testing RiskAnalyzer...")

    # Create sample returns
    returns = pd.Series(np.random.randn(500) * 0.01)
    returns_df = pd.DataFrame(
        np.random.randn(500, 3) * 0.01,
        columns=['A', 'B', 'C']
    )

    risk_analyzer = RiskAnalyzer()

    # Test GARCH(1,1)
    garch_result = risk_analyzer.estimate_garch_11(returns)
    assert 'omega' in garch_result, "GARCH missing omega"
    assert 'alpha' in garch_result, "GARCH missing alpha"
    assert 'beta' in garch_result, "GARCH missing beta"
    assert garch_result['alpha'] >= 0, "Alpha should be non-negative"
    assert garch_result['beta'] >= 0, "Beta should be non-negative"

    # Test VaR
    for method in ['parametric', 'historical', 'monte_carlo']:
        var_result = risk_analyzer.calculate_var(returns, method=method)
        assert 'VaR' in var_result, f"VaR missing for method {method}"
        assert var_result['VaR'] > 0, "VaR should be positive"
        assert 'Expected_Shortfall' in var_result, "Missing Expected Shortfall"

    # Test correlation matrix
    corr_analysis = risk_analyzer.correlation_matrix_analysis(returns_df)
    assert 'correlation_matrix' in corr_analysis, "Missing correlation matrix"
    assert 'covariance_matrix' in corr_analysis, "Missing covariance matrix"
    assert 'eigenvalues' in corr_analysis, "Missing eigenvalues"

    print("  ✓ All RiskAnalyzer tests passed")
    return True


def test_portfolio_optimizer():
    """Test Portfolio Optimizer module"""
    print("\n[TEST 4] Testing PortfolioOptimizer...")

    # Create sample data
    expected_returns = np.array([0.10, 0.12, 0.08, 0.15])
    cov_matrix = np.array([
        [0.04, 0.01, 0.01, 0.02],
        [0.01, 0.05, 0.01, 0.01],
        [0.01, 0.01, 0.03, 0.01],
        [0.02, 0.01, 0.01, 0.06]
    ])

    optimizer = PortfolioOptimizer()

    # Test mean-variance optimization
    mv_result = optimizer.mean_variance_optimization(expected_returns, cov_matrix)
    assert 'weights' in mv_result, "Missing weights"
    assert abs(np.sum(mv_result['weights']) - 1.0) < 1e-6, "Weights don't sum to 1"
    assert all(mv_result['weights'] >= -1e-6), "Weights should be non-negative"
    assert 'sharpe_ratio' in mv_result, "Missing Sharpe ratio"

    # Test efficient frontier
    efficient_frontier = optimizer.efficient_frontier(expected_returns, cov_matrix, n_points=10)
    assert len(efficient_frontier) > 0, "Efficient frontier is empty"
    assert 'return' in efficient_frontier.columns, "Missing return column"
    assert 'volatility' in efficient_frontier.columns, "Missing volatility column"

    # Test Kelly Criterion
    kelly_fraction = optimizer.kelly_criterion(0.55, 0.02, -0.015)
    assert 0 <= kelly_fraction <= 1, "Kelly fraction out of valid range"

    # Test Risk Parity
    rp_weights = optimizer.risk_parity(cov_matrix)
    assert len(rp_weights) == len(expected_returns), "Risk parity weights length mismatch"
    assert abs(np.sum(rp_weights) - 1.0) < 1e-6, "Risk parity weights don't sum to 1"

    print("  ✓ All PortfolioOptimizer tests passed")
    return True


def test_execution_analyzer():
    """Test Execution Analyzer module"""
    print("\n[TEST 5] Testing ExecutionAnalyzer...")

    executor = ExecutionAnalyzer()

    # Test Monte Carlo simulation
    mc_result = executor.monte_carlo_simulation(
        mu=0.10, sigma=0.20, S0=100, T=1.0, n_simulations=1000
    )
    assert 'paths' in mc_result, "Missing simulation paths"
    assert 'final_prices' in mc_result, "Missing final prices"
    assert len(mc_result['final_prices']) == 1000, "Wrong number of simulations"
    assert mc_result['mean_final_price'] > 0, "Mean final price should be positive"

    # Test market impact
    impact = executor.calculate_market_impact(
        order_size=100000,
        average_daily_volume=1000000,
        volatility=0.20
    )
    assert 'temporary_impact' in impact, "Missing temporary impact"
    assert 'permanent_impact' in impact, "Missing permanent impact"
    assert 'total_impact' in impact, "Missing total impact"
    assert impact['total_impact'] >= 0, "Total impact should be non-negative"

    # Test backtesting
    returns = pd.Series(np.random.randn(100) * 0.01)
    signals = pd.Series(np.random.choice([-1, 0, 1], 100))

    backtest_result = executor.backtest_strategy(signals, returns)
    assert 'total_return' in backtest_result, "Missing total return"
    assert 'sharpe_ratio' in backtest_result, "Missing Sharpe ratio"
    assert 'max_drawdown' in backtest_result, "Missing max drawdown"
    assert 'win_rate' in backtest_result, "Missing win rate"

    print("  ✓ All ExecutionAnalyzer tests passed")
    return True


def test_full_system():
    """Test the complete Stock Analysis System"""
    print("\n[TEST 6] Testing Complete System...")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.DataFrame({
        'A': 100 + np.cumsum(np.random.randn(252) * 1),
        'B': 150 + np.cumsum(np.random.randn(252) * 1.5),
        'C': 80 + np.cumsum(np.random.randn(252) * 0.8)
    }, index=dates)

    market_caps = pd.Series([10e9, 5e9, 50e9], index=['A', 'B', 'C'])
    book_to_market = pd.Series([0.8, 1.2, 0.5], index=['A', 'B', 'C'])

    # Run full analysis
    system = StockAnalysisSystem()
    results = system.analyze_stock(prices, market_caps, book_to_market)

    # Check all major components
    assert 'preprocessed_data' in results, "Missing preprocessed data"
    assert 'risk_analysis' in results, "Missing risk analysis"
    assert 'portfolio_optimization' in results, "Missing portfolio optimization"
    assert 'factor_analysis' in results, "Missing factor analysis"
    assert 'monte_carlo' in results, "Missing Monte Carlo results"

    # Check specific results
    assert 'mean_variance' in results['portfolio_optimization'], "Missing MV optimization"
    assert 'var' in results['risk_analysis'], "Missing VaR analysis"

    print("  ✓ All Complete System tests passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 Running Stock Analysis System Tests")
    print("=" * 70)

    tests = [
        test_data_preprocessor,
        test_factor_model,
        test_risk_analyzer,
        test_portfolio_optimizer,
        test_execution_analyzer,
        test_full_system
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ All tests passed! The system is working correctly.")
        print("\n📝 Next steps:")
        print("  1. Run 'python quick_start.py' for a simple example")
        print("  2. Run 'python example_usage.py' for detailed examples")
        print("  3. Read README.md for complete documentation")
    else:
        print("\n❌ Some tests failed. Please check your installation.")
        print("  Try: pip install -r requirements.txt")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
