import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Simulate Data
np.random.seed(42)
num_assets = 5
num_days = 252

# Simulated daily returns for 5 assets
returns = np.random.randn(num_days, num_assets) * 0.01
dates = pd.date_range(start="2023-01-01", periods=num_days)
assets = [f"Asset_{i+1}" for i in range(num_assets)]
price_df = pd.DataFrame(100 + np.cumsum(returns, axis=0), index=dates, columns=assets)

# Step 2: Signal Generation (SMA-Based)
short_window = 20
long_window = 50

# Calculate SMAs
short_sma = price_df.rolling(window=short_window).mean()
long_sma = price_df.rolling(window=long_window).mean()

# Generate signals: positive if short SMA > long SMA, negative otherwise
sma_signal = (short_sma.iloc[-1] - long_sma.iloc[-1])
sma_signal /= np.linalg.norm(sma_signal)  # Normalize signals

# Step 3: Covariance Matrix
returns_df = price_df.pct_change().dropna()  # Daily returns
cov_matrix = returns_df.cov()

# Step 4: Markowitz Optimization
def portfolio_variance(weights, cov_matrix):
    """Calculate portfolio variance."""
    return weights.T @ cov_matrix @ weights

def optimize_portfolio(expected_returns, cov_matrix):
    """Optimize portfolio using Markowitz framework."""
    num_assets = len(expected_returns)
    init_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1)] * num_assets  # Long-only
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum of weights = 1
    ]

    result = minimize(
        fun=portfolio_variance,
        x0=init_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x if result.success else None

# Step 5: Portfolio Construction
optimized_weights = optimize_portfolio(sma_signal, cov_matrix)

# Display Results
print("SMA Signals (Normalized):")
print(sma_signal)
print("\nOptimized Portfolio Weights:")
print(optimized_weights)

# Step 6: Portfolio Performance
# Compute expected portfolio return and variance
expected_portfolio_return = np.dot(optimized_weights, sma_signal)
expected_portfolio_variance = portfolio_variance(optimized_weights, cov_matrix)

print("\nExpected Portfolio Return:", expected_portfolio_return)
print("Expected Portfolio Variance:", expected_portfolio_variance)

# Optional: Visualize Results
plt.figure(figsize=(10, 6))
plt.bar(assets, optimized_weights)
plt.title("Optimized Portfolio Weights Based on SMA Signals")
plt.ylabel("Weight")
plt.xlabel("Assets")
plt.show()
