"""
Volatility Trading Book - Code Examples
Chapter 2: Measuring Volatility

This module contains implementations of various volatility calculation methods
discussed in the book, including historical volatility estimators and
volatility forecasting models.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import warnings

class VolatilityCalculator:
    """
    A comprehensive volatility calculator implementing various methods
    from academic literature and industry practice.
    """
    
    def __init__(self, annualization_factor: int = 252):
        """
        Initialize the volatility calculator.
        
        Args:
            annualization_factor: Number of trading days per year for annualization
        """
        self.annualization_factor = annualization_factor
    
    def close_to_close_volatility(self, prices: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate close-to-close volatility (simplest method).
        
        Args:
            prices: Array of closing prices
            
        Returns:
            Annualized volatility
        """
        returns = np.log(prices[1:] / prices[:-1])
        return np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)
    
    def parkinson_volatility(self, high: np.ndarray, low: np.ndarray) -> float:
        """
        Calculate Parkinson volatility estimator using high and low prices.
        More efficient than close-to-close for continuous trading.
        
        Args:
            high: Array of daily high prices
            low: Array of daily low prices
            
        Returns:
            Annualized volatility
        """
        hl_ratio = np.log(high / low)
        parkinson_var = np.mean(hl_ratio ** 2) / (4 * np.log(2))
        return np.sqrt(parkinson_var * self.annualization_factor)
    
    def garman_klass_volatility(self, open_: np.ndarray, high: np.ndarray, 
                               low: np.ndarray, close: np.ndarray) -> float:
        """
        Calculate Garman-Klass volatility estimator using OHLC prices.
        
        Args:
            open_: Array of opening prices
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            
        Returns:
            Annualized volatility
        """
        ln_ho = np.log(high / open_)
        ln_hc = np.log(high / close)
        ln_lo = np.log(low / open_)
        ln_lc = np.log(low / close)
        
        gk_var = np.mean(ln_hc * ln_ho + ln_lc * ln_lo)
        return np.sqrt(gk_var * self.annualization_factor)
    
    def yang_zhang_volatility(self, open_: np.ndarray, high: np.ndarray, 
                             low: np.ndarray, close: np.ndarray) -> float:
        """
        Calculate Yang-Zhang volatility estimator.
        Most sophisticated classical estimator handling overnight jumps.
        
        Args:
            open_: Array of opening prices
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            
        Returns:
            Annualized volatility
        """
        # Overnight returns
        overnight = np.log(open_[1:] / close[:-1])
        
        # Open-to-close returns
        open_to_close = np.log(close / open_)
        
        # Rogers-Satchell component
        rs = np.log(high / close) * np.log(high / open_) + \
             np.log(low / close) * np.log(low / open_)
        
        # Calculate components
        overnight_var = np.var(overnight, ddof=1)
        oc_var = np.var(open_to_close, ddof=1)
        rs_var = np.mean(rs)
        
        # Combine components
        k = 0.34 / (1 + (len(close) + 1) / (len(close) - 1))
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * self.annualization_factor)
    
    def ewma_volatility(self, returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
        """
        Calculate Exponentially Weighted Moving Average volatility.
        
        Args:
            returns: Array of returns
            lambda_: Decay factor (RiskMetrics uses 0.94)
            
        Returns:
            Array of EWMA volatilities
        """
        var_ewma = np.zeros(len(returns))
        var_ewma[0] = returns[0] ** 2
        
        for t in range(1, len(returns)):
            var_ewma[t] = lambda_ * var_ewma[t-1] + (1 - lambda_) * returns[t-1] ** 2
        
        return np.sqrt(var_ewma * self.annualization_factor)


class VolatilityCone:
    """
    Implementation of volatility cone analysis for putting current
    volatility in historical context.
    """
    
    def __init__(self, prices: pd.Series, horizons: List[int] = None):
        """
        Initialize volatility cone analysis.
        
        Args:
            prices: Time series of prices
            horizons: List of horizons to analyze (in days)
        """
        self.prices = prices
        self.horizons = horizons or [10, 20, 30, 60, 90, 120, 180, 252]
        self.calculator = VolatilityCalculator()
        
    def calculate_cone(self) -> pd.DataFrame:
        """
        Calculate volatility cone across all horizons.
        
        Returns:
            DataFrame with percentile volatilities for each horizon
        """
        cone_data = {}
        
        for horizon in self.horizons:
            vols = []
            
            # Calculate rolling volatility for each period
            for i in range(horizon, len(self.prices)):
                period_prices = self.prices.iloc[i-horizon:i]
                vol = self.calculator.close_to_close_volatility(period_prices)
                vols.append(vol)
            
            # Calculate percentiles
            vols = np.array(vols)
            percentiles = [0, 5, 25, 50, 75, 95, 100]
            cone_data[horizon] = [np.percentile(vols, p) for p in percentiles]
        
        # Create DataFrame
        cone_df = pd.DataFrame(cone_data, 
                              index=['Min', '5th', '25th', '50th', '75th', '95th', 'Max'])
        return cone_df.T
    
    def current_volatility_percentile(self, current_vol: float, horizon: int) -> float:
        """
        Determine where current volatility sits in historical distribution.
        
        Args:
            current_vol: Current volatility level
            horizon: Time horizon for comparison
            
        Returns:
            Percentile rank of current volatility
        """
        # Calculate historical volatilities for the horizon
        vols = []
        for i in range(horizon, len(self.prices)):
            period_prices = self.prices.iloc[i-horizon:i]
            vol = self.calculator.close_to_close_volatility(period_prices)
            vols.append(vol)
        
        # Calculate percentile rank
        percentile = stats.percentileofscore(vols, current_vol)
        return percentile
    
    def plot_cone(self, title: str = "Volatility Cone") -> plt.Figure:
        """
        Plot the volatility cone.
        
        Args:
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        cone_df = self.calculate_cone()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot percentile bands
        ax.fill_between(cone_df.index, cone_df['5th'], cone_df['95th'], 
                       alpha=0.2, color='blue', label='5th-95th percentile')
        ax.fill_between(cone_df.index, cone_df['25th'], cone_df['75th'], 
                       alpha=0.3, color='blue', label='25th-75th percentile')
        
        # Plot median line
        ax.plot(cone_df.index, cone_df['50th'], 'b-', linewidth=2, label='Median')
        
        # Plot min/max
        ax.plot(cone_df.index, cone_df['Min'], 'r--', alpha=0.5, label='Min/Max')
        ax.plot(cone_df.index, cone_df['Max'], 'r--', alpha=0.5)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Volatility (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class GARCHModel:
    """
    Simple GARCH(1,1) implementation for volatility forecasting.
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
    
    def fit(self, returns: np.ndarray, max_iter: int = 1000) -> dict:
        """
        Fit GARCH(1,1) model using Maximum Likelihood Estimation.
        
        Args:
            returns: Array of returns
            max_iter: Maximum iterations for optimization
            
        Returns:
            Dictionary with fitted parameters
        """
        from scipy.optimize import minimize
        
        def garch_likelihood(params, returns):
            """Calculate negative log-likelihood for GARCH(1,1)"""
            omega, alpha, beta = params
            
            # Check parameter constraints
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
                return 1e8
            
            # Initialize variance
            var = np.zeros(len(returns))
            var[0] = np.var(returns)
            
            # Calculate conditional variances
            for t in range(1, len(returns)):
                var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
            
            # Calculate log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + returns**2 / var)
            return -log_likelihood
        
        # Initial parameter guess
        initial_params = [0.01, 0.1, 0.85]
        
        # Optimize
        result = minimize(garch_likelihood, initial_params, 
                         args=(returns,), method='L-BFGS-B',
                         bounds=[(1e-6, 1), (1e-6, 0.5), (1e-6, 0.99)])
        
        if result.success:
            self.params = {
                'omega': result.x[0],
                'alpha': result.x[1], 
                'beta': result.x[2]
            }
            self.fitted = True
            return self.params
        else:
            raise RuntimeError("GARCH optimization failed")
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Generate volatility forecasts using fitted GARCH model.
        
        Args:
            returns: Historical returns for initialization
            horizon: Number of periods to forecast
            
        Returns:
            Array of volatility forecasts
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        omega, alpha, beta = self.params.values()
        
        # Calculate current conditional variance
        current_var = omega + alpha * returns[-1]**2 + beta * np.var(returns)
        
        # Generate forecasts
        forecasts = np.zeros(horizon)
        long_run_var = omega / (1 - alpha - beta)
        
        for h in range(horizon):
            if h == 0:
                forecasts[h] = current_var
            else:
                # Multi-step ahead forecast
                forecasts[h] = long_run_var + (alpha + beta)**h * (current_var - long_run_var)
        
        return np.sqrt(forecasts * 252)  # Annualized volatility


# Example usage and testing functions
def example_volatility_calculations():
    """
    Example showing different volatility calculation methods.
    """
    # Generate sample data
    np.random.seed(42)
    n_days = 252
    returns = np.random.normal(0, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some intraday data
    high = prices * np.random.uniform(1.00, 1.05, n_days)
    low = prices * np.random.uniform(0.95, 1.00, n_days) 
    open_ = prices * np.random.uniform(0.98, 1.02, n_days)
    
    # Calculate volatilities
    calc = VolatilityCalculator()
    
    cc_vol = calc.close_to_close_volatility(prices)
    parkinson_vol = calc.parkinson_volatility(high, low)
    gk_vol = calc.garman_klass_volatility(open_, high, low, prices)
    yz_vol = calc.yang_zhang_volatility(open_, high, low, prices)
    
    print("Volatility Estimation Results:")
    print(f"Close-to-Close: {cc_vol:.2%}")
    print(f"Parkinson: {parkinson_vol:.2%}")
    print(f"Garman-Klass: {gk_vol:.2%}")
    print(f"Yang-Zhang: {yz_vol:.2%}")
    
    return {
        'close_to_close': cc_vol,
        'parkinson': parkinson_vol,
        'garman_klass': gk_vol,
        'yang_zhang': yz_vol
    }


def example_volatility_cone():
    """
    Example showing volatility cone analysis.
    """
    # Generate sample price data
    np.random.seed(42)
    n_days = 1000
    returns = np.random.normal(0, 0.015, n_days)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    
    # Create volatility cone
    cone = VolatilityCone(prices)
    cone_df = cone.calculate_cone()
    
    print("\nVolatility Cone Analysis:")
    print(cone_df.round(4))
    
    # Current volatility percentile
    current_vol = 0.20
    percentile = cone.current_volatility_percentile(current_vol, 30)
    print(f"\nCurrent 30-day volatility of {current_vol:.1%} is at {percentile:.1f}th percentile")
    
    return cone_df


def example_garch_forecasting():
    """
    Example showing GARCH volatility forecasting.
    """
    # Generate sample returns with volatility clustering
    np.random.seed(42)
    n_days = 500
    
    # Create returns with GARCH properties
    returns = np.zeros(n_days)
    var = np.zeros(n_days)
    var[0] = 0.01
    
    omega, alpha, beta = 0.01, 0.1, 0.85
    
    for t in range(1, n_days):
        var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
        returns[t] = np.sqrt(var[t]) * np.random.normal()
    
    # Fit GARCH model
    garch = GARCHModel()
    params = garch.fit(returns)
    
    print("\nGARCH(1,1) Parameter Estimates:")
    for param, value in params.items():
        print(f"{param}: {value:.6f}")
    
    # Generate forecasts
    forecasts = garch.forecast(returns, horizon=10)
    
    print("\nVolatility Forecasts (10 days ahead):")
    for i, forecast in enumerate(forecasts, 1):
        print(f"Day {i}: {forecast:.2%}")
    
    return params, forecasts


if __name__ == "__main__":
    print("Running Volatility Trading Book Examples")
    print("=" * 50)
    
    # Run examples
    vol_results = example_volatility_calculations()
    cone_results = example_volatility_cone()
    garch_results = example_garch_forecasting()
    
    print("\nAll examples completed successfully!")