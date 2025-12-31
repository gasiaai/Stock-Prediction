"""
Stock Analysis System
ระบบวิเคราะห์หุ้นแบบครบวงจร
Based on MIT Financial Engineering principles
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.linalg import qr
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    1. การเตรียมข้อมูลและการประมวลผลเบื้องต้น (Data Pre-processing)
    - Log Returns transformation
    - Outlier detection and handling
    - QR Decomposition for fast computation
    """

    def __init__(self):
        self.scaler_params = {}

    def to_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        แปลงราคาหุ้นเป็น Log Returns
        Log Return = ln(P_t / P_{t-1})
        """
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.dropna()

    def detect_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        ตรวจจับค่าผิดปกติ (Outliers) โดยใช้ Z-score
        threshold: จำนวน standard deviations
        """
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        # Convert to DataFrame to use fillna
        z_scores_df = pd.DataFrame(z_scores, index=data.index, columns=data.columns)
        z_scores_df = z_scores_df.fillna(0)  # Fill NaN with 0 to avoid indexing issues
        outlier_mask = z_scores_df > threshold

        print(f"พบ Outliers: {outlier_mask.sum().sum()} จุด")
        return outlier_mask

    def handle_missing_data(self, data: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        จัดการข้อมูลที่ขาดหาย (Missing Data)
        method: 'linear' (Linear Interpolation) หรือ 'brownian' (Brownian Bridge)
        """
        if method == 'linear':
            return data.interpolate(method='linear', limit_direction='both')
        elif method == 'brownian':
            # Brownian Bridge interpolation
            return data.interpolate(method='quadratic', limit_direction='both')
        else:
            return data.fillna(method='ffill').fillna(method='bfill')

    def qr_decomposition(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR Decomposition สำหรับการคำนวณ Least Squares อย่างรวดเร็ว
        X = QR where Q is orthogonal and R is upper triangular
        """
        Q, R = qr(X, mode='economic')
        return Q, R

    def preprocess_pipeline(self, prices: pd.DataFrame,
                          remove_outliers: bool = True,
                          outlier_threshold: float = 3.0) -> Dict:
        """
        Pipeline การประมวลผลข้อมูลแบบครบวงจร
        """
        # 1. Convert to log returns
        log_returns = self.to_log_returns(prices)

        # 2. Handle missing data
        log_returns = self.handle_missing_data(log_returns)

        # 3. Detect and handle outliers
        if remove_outliers:
            outlier_mask = self.detect_outliers(log_returns, outlier_threshold)
            # Replace outliers with median
            for col in log_returns.columns:
                log_returns.loc[outlier_mask[col], col] = log_returns[col].median()

        # 4. QR decomposition for future use
        X = log_returns.values
        Q, R = self.qr_decomposition(X)

        return {
            'log_returns': log_returns,
            'Q': Q,
            'R': R,
            'raw_prices': prices
        }


class FactorModel:
    """
    2. แบบจำลองปัจจัยและการหาค่า Alpha (Alpha & Factor Modeling)
    - Fama-French factor model
    - Regression analysis for Beta calculation
    - Machine Learning pattern detection (Kernel Methods)
    """

    def __init__(self):
        self.betas = {}
        self.alphas = {}
        self.factor_loadings = {}

    def create_fama_french_factors(self, returns: pd.DataFrame,
                                   market_caps: pd.Series,
                                   book_to_market: pd.Series) -> pd.DataFrame:
        """
        สร้างปัจจัย Fama-French
        - SMB (Small Minus Big): Size factor
        - HML (High Minus Low): Value factor
        - MKT (Market factor)
        """
        # Market factor (equal-weighted market return)
        mkt_factor = returns.mean(axis=1)

        # Size factor (SMB)
        median_size = market_caps.median()
        small_stocks = market_caps[market_caps <= median_size].index
        big_stocks = market_caps[market_caps > median_size].index

        smb_factor = returns[small_stocks].mean(axis=1) - returns[big_stocks].mean(axis=1)

        # Value factor (HML)
        median_btm = book_to_market.median()
        high_btm = book_to_market[book_to_market >= median_btm].index
        low_btm = book_to_market[book_to_market < median_btm].index

        hml_factor = returns[high_btm].mean(axis=1) - returns[low_btm].mean(axis=1)

        factors = pd.DataFrame({
            'MKT': mkt_factor,
            'SMB': smb_factor,
            'HML': hml_factor
        })

        return factors

    def regression_analysis(self, stock_returns: pd.Series,
                          factors: pd.DataFrame) -> Dict:
        """
        การวิเคราะห์การถดถอย (Regression Analysis)
        R_i = α + β₁*MKT + β₂*SMB + β₃*HML + ε
        """
        # Prepare data
        y = stock_returns.values
        X = factors.values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept

        # OLS estimation: β = (X'X)^(-1)X'y
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            beta_hat = np.linalg.pinv(X) @ y

        # Calculate residuals
        y_pred = X @ beta_hat
        residuals = y - y_pred

        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate standard errors (only if matrix is invertible)
        try:
            mse = ss_res / (len(y) - len(beta_hat))
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            t_stats = beta_hat / se_beta
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(beta_hat)))
        except np.linalg.LinAlgError:
            # If singular, set to NaN
            se_beta = np.full(len(beta_hat), np.nan)
            t_stats = np.full(len(beta_hat), np.nan)
            p_values = np.full(len(beta_hat), np.nan)

        return {
            'alpha': beta_hat[0],
            'betas': beta_hat[1:],
            'r_squared': r_squared,
            'residuals': residuals,
            't_stats': t_stats,
            'p_values': p_values,
            'factor_names': factors.columns.tolist()
        }

    def kernel_regression(self, X: np.ndarray, y: np.ndarray,
                         bandwidth: float = 1.0) -> callable:
        """
        Kernel Methods สำหรับการตรวจจับรูปแบบที่ซับซ้อน (Non-linear patterns)
        ใช้ Gaussian Kernel
        """
        def gaussian_kernel(x1, x2, h):
            return np.exp(-np.sum((x1 - x2)**2) / (2 * h**2))

        def predict(x_new):
            weights = np.array([gaussian_kernel(x_new, x_i, bandwidth) for x_i in X])
            weights /= weights.sum()
            return np.dot(weights, y)

        return predict


class RiskAnalyzer:
    """
    3. การวิเคราะห์ความผันผวนและความเสี่ยง (Volatility & Risk Analysis)
    - GARCH(1,1) model
    - Value at Risk (VaR)
    - Correlation Matrix analysis
    """

    def __init__(self):
        self.garch_params = {}
        self.var_values = {}

    def estimate_garch_11(self, returns: pd.Series,
                         max_iter: int = 1000) -> Dict:
        """
        แบบจำลอง GARCH(1,1)
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        """
        # Initial variance estimate
        variance = returns.var()

        # Parameters: [omega, alpha, beta]
        def garch_likelihood(params):
            omega, alpha, beta = params

            # Constraint: alpha + beta < 1 for stationarity
            if alpha + beta >= 1 or alpha < 0 or beta < 0 or omega < 0:
                return 1e10

            T = len(returns)
            sigma2 = np.zeros(T)
            sigma2[0] = variance

            for t in range(1, T):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]

            # Log-likelihood (assuming normal distribution)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
            return -log_likelihood  # Minimize negative log-likelihood

        # Optimize
        initial_params = [variance * 0.01, 0.05, 0.9]
        result = optimize.minimize(garch_likelihood, initial_params,
                                  method='L-BFGS-B',
                                  bounds=[(1e-6, None), (0, 1), (0, 1)])

        omega, alpha, beta = result.x

        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': alpha + beta,
            'unconditional_variance': omega / (1 - alpha - beta) if alpha + beta < 1 else None
        }

    def calculate_var(self, returns: pd.Series,
                     confidence_level: float = 0.99,
                     method: str = 'parametric') -> Dict:
        """
        Value at Risk (VaR)
        confidence_level: 0.95 (1.645 SD) or 0.99 (2.33 SD)
        """
        if method == 'parametric':
            # Assume normal distribution
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)

        elif method == 'historical':
            # Historical simulation
            var = -np.percentile(returns, (1 - confidence_level) * 100)

        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = returns.mean()
            std = returns.std()
            simulations = np.random.normal(mean, std, 10000)
            var = -np.percentile(simulations, (1 - confidence_level) * 100)

        # Expected Shortfall (CVaR)
        es = -returns[returns <= -var].mean()

        return {
            'VaR': var,
            'Expected_Shortfall': es,
            'confidence_level': confidence_level,
            'method': method
        }

    def correlation_matrix_analysis(self, returns: pd.DataFrame) -> Dict:
        """
        เมทริกซ์ความสัมพันธ์ (Correlation Matrix)
        Portfolio Variance = X^T * Σ * X
        """
        # Correlation and covariance matrices
        corr_matrix = returns.corr()
        cov_matrix = returns.cov()

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return {
            'correlation_matrix': corr_matrix,
            'covariance_matrix': cov_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': eigenvalues / eigenvalues.sum()
        }


class PortfolioOptimizer:
    """
    4. การจัดการพอร์ตโฟลิโอและขนาดการลงทุน (Portfolio & Position Sizing)
    - Mean-Variance Optimization (Markowitz)
    - Kelly Criterion
    - Risk Parity
    """

    def __init__(self):
        self.optimal_weights = {}

    def mean_variance_optimization(self, expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   target_return: Optional[float] = None,
                                   risk_free_rate: float = 0.02) -> Dict:
        """
        Mean-Variance Optimization (Markowitz)
        Minimize: w^T * Σ * w
        Subject to: w^T * μ = target_return, Σw = 1
        """
        n_assets = len(expected_returns)

        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w.T @ expected_returns - target_return
            })

        # Bounds: 0 <= w <= 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        result = optimize.minimize(portfolio_variance, initial_weights,
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)

        optimal_weights = result.x
        portfolio_return = optimal_weights.T @ expected_returns
        portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }

    def efficient_frontier(self, expected_returns: np.ndarray,
                          cov_matrix: np.ndarray,
                          n_points: int = 50) -> pd.DataFrame:
        """
        คำนวณ Efficient Frontier
        """
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)

        efficient_portfolios = []

        for target in target_returns:
            try:
                result = self.mean_variance_optimization(
                    expected_returns, cov_matrix, target_return=target
                )
                efficient_portfolios.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            except:
                continue

        return pd.DataFrame(efficient_portfolios)

    def kelly_criterion(self, win_prob: float,
                       win_return: float,
                       loss_return: float) -> float:
        """
        Kelly Criterion สำหรับ Position Sizing
        f* = (p*b - q) / b
        where: p = win probability, q = 1-p, b = win/loss ratio
        """
        q = 1 - win_prob
        b = abs(win_return / loss_return)

        kelly_fraction = (win_prob * b - q) / b

        # Safety: use half-Kelly or quarter-Kelly
        return max(0, min(kelly_fraction * 0.5, 0.25))  # Conservative approach

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk Parity Portfolio
        Equal risk contribution from each asset
        """
        n_assets = cov_matrix.shape[0]

        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # We want equal risk contribution
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk)**2)

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))

        initial_weights = np.array([1/n_assets] * n_assets)

        result = optimize.minimize(risk_contribution, initial_weights,
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)

        return result.x


class ExecutionAnalyzer:
    """
    5. การนำไปใช้จริงและข้อจำกัดทางเทคนิค (Execution & Constraints)
    - Monte Carlo Simulation
    - Market Impact และ Slippage
    - Short-term Trading Strategy
    """

    def __init__(self):
        self.simulation_results = {}

    def monte_carlo_simulation(self, mu: float, sigma: float,
                              S0: float, T: float,
                              n_simulations: int = 10000,
                              n_steps: int = 252) -> Dict:
        """
        การจำลอง Monte Carlo สำหรับราคาหุ้น
        Geometric Brownian Motion: dS = μ*S*dt + σ*S*dW
        """
        dt = T / n_steps

        # Generate random paths
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S0

        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

        final_prices = paths[:, -1]

        return {
            'paths': paths,
            'final_prices': final_prices,
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'std_final_price': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95)
        }

    def calculate_market_impact(self, order_size: float,
                               average_daily_volume: float,
                               volatility: float,
                               impact_coefficient: float = 0.1) -> Dict:
        """
        คำนวณ Market Impact
        Impact ≈ σ * (Q/V)^0.5
        where Q = order size, V = average daily volume, σ = volatility
        """
        participation_rate = order_size / average_daily_volume

        # Square-root model
        temporary_impact = impact_coefficient * volatility * np.sqrt(participation_rate)

        # Permanent impact (smaller)
        permanent_impact = 0.5 * temporary_impact

        total_impact = temporary_impact + permanent_impact

        return {
            'temporary_impact': temporary_impact,
            'permanent_impact': permanent_impact,
            'total_impact': total_impact,
            'participation_rate': participation_rate,
            'estimated_slippage_bps': total_impact * 10000  # in basis points
        }

    def backtest_strategy(self, signals: pd.Series,
                         returns: pd.Series,
                         transaction_cost: float = 0.001) -> Dict:
        """
        Backtesting สำหรับกลยุทธ์ระยะสั้น
        signals: 1 (long), -1 (short), 0 (neutral)
        """
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns

        # Account for transaction costs
        position_changes = signals.diff().abs()
        costs = position_changes * transaction_cost
        strategy_returns_net = strategy_returns - costs

        # Performance metrics
        cumulative_returns = (1 + strategy_returns_net).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Sharpe ratio (annualized)
        sharpe_ratio = np.sqrt(252) * strategy_returns_net.mean() / strategy_returns_net.std()

        # Maximum drawdown
        cummax = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        wins = strategy_returns_net > 0
        win_rate = wins.sum() / len(strategy_returns_net)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'cumulative_returns': cumulative_returns,
            'strategy_returns': strategy_returns_net
        }


class StockAnalysisSystem:
    """
    ระบบวิเคราะห์หุ้นแบบครบวงจร
    รวมทุกโมดูลเข้าด้วยกัน
    """

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.factor_model = FactorModel()
        self.risk_analyzer = RiskAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.execution_analyzer = ExecutionAnalyzer()

    def analyze_stock(self, prices: pd.DataFrame,
                     market_caps: Optional[pd.Series] = None,
                     book_to_market: Optional[pd.Series] = None) -> Dict:
        """
        วิเคราะห์หุ้นแบบครบวงจร
        """
        print("=" * 60)
        print("เริ่มการวิเคราะห์หุ้น - Stock Analysis System")
        print("=" * 60)

        # 1. Data Preprocessing
        print("\n[1/5] กำลังประมวลผลข้อมูล...")
        preprocessed = self.preprocessor.preprocess_pipeline(prices)
        log_returns = preprocessed['log_returns']
        print(f"✓ ประมวลผลข้อมูล {len(log_returns)} วัน สำเร็จ")

        # 2. Risk Analysis
        print("\n[2/5] กำลังวิเคราะห์ความเสี่ยง...")
        risk_results = {}

        # Correlation matrix
        corr_analysis = self.risk_analyzer.correlation_matrix_analysis(log_returns)
        risk_results['correlation'] = corr_analysis

        # VaR for each stock
        var_results = {}
        for col in log_returns.columns:
            var_results[col] = self.risk_analyzer.calculate_var(
                log_returns[col], confidence_level=0.99
            )
        risk_results['var'] = var_results
        print(f"✓ วิเคราะห์ความเสี่ยงสำเร็จ")

        # 3. Portfolio Optimization
        print("\n[3/5] กำลังหาพอร์ตโฟลิโอที่เหมาะสม...")
        expected_returns = log_returns.mean() * 252  # Annualized
        cov_matrix = log_returns.cov() * 252  # Annualized

        # Mean-Variance Optimization
        mv_result = self.portfolio_optimizer.mean_variance_optimization(
            expected_returns.values, cov_matrix.values
        )

        # Risk Parity
        rp_weights = self.portfolio_optimizer.risk_parity(cov_matrix.values)

        portfolio_results = {
            'mean_variance': mv_result,
            'risk_parity_weights': rp_weights,
            'efficient_frontier': self.portfolio_optimizer.efficient_frontier(
                expected_returns.values, cov_matrix.values
            )
        }
        print(f"✓ คำนวณพอร์ตโฟลิโอที่เหมาะสมสำเร็จ")

        # 4. Factor Analysis (if data available)
        factor_results = None
        if market_caps is not None and book_to_market is not None:
            print("\n[4/5] กำลังวิเคราะห์ปัจจัย...")
            factors = self.factor_model.create_fama_french_factors(
                log_returns, market_caps, book_to_market
            )

            factor_results = {}
            for col in log_returns.columns:
                factor_results[col] = self.factor_model.regression_analysis(
                    log_returns[col], factors
                )
            print(f"✓ วิเคราะห์ปัจจัยสำเร็จ")
        else:
            print("\n[4/5] ข้าม Factor Analysis (ไม่มีข้อมูล market cap)")

        # 5. Monte Carlo Simulation
        print("\n[5/5] กำลังจำลอง Monte Carlo...")
        simulation_results = {}
        for col in log_returns.columns[:3]:  # First 3 stocks for demo
            mu = expected_returns[col]
            sigma = log_returns[col].std() * np.sqrt(252)
            S0 = prices[col].iloc[-1]

            simulation_results[col] = self.execution_analyzer.monte_carlo_simulation(
                mu, sigma, S0, T=1.0, n_simulations=1000
            )
        print(f"✓ จำลอง Monte Carlo สำเร็จ")

        print("\n" + "=" * 60)
        print("การวิเคราะห์เสร็จสมบูรณ์!")
        print("=" * 60)

        return {
            'preprocessed_data': preprocessed,
            'risk_analysis': risk_results,
            'portfolio_optimization': portfolio_results,
            'factor_analysis': factor_results,
            'monte_carlo': simulation_results
        }


if __name__ == "__main__":
    print("Stock Analysis System - ระบบวิเคราะห์หุ้น")
    print("กรุณาดู example_usage.py สำหรับตัวอย่างการใช้งาน")
