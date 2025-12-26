import numpy as np
import scipy.stats as stats

def adjusted_sharpe_ratio(returns):
    """
    Calculates the Adjusted Sharpe Ratio using the Pezier-White approximation.
    
    ASR = SR * [1 + (S/6)*SR - ((K-3)/24)*SR^2]
    
    Where:
    SR = Annualized Sharpe Ratio
    S = Skewness
    K = Kurtosis (Fisher)
    
    Args:
        returns (np.array): Array of returns (usually excess returns).
    
    Returns:
        float: The adjusted Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate moments
    mean = np.mean(returns)
    std = np.std(returns)
    
    if std == 0:
        return 0.0
        
    sr = mean / std
    
    # Skewness
    s = stats.skew(returns)
    
    # Kurtosis (Fisher, so normal is 0. Scipy defaults to Fisher=True but let's be explicit)
    # stats.kurtosis returns excess kurtosis (K - 3) by default.
    # Pezier-White formula uses K (raw kurtosis) or K-3 (excess kurtosis)?
    # The formula usually cited is:
    # ASR = SR * (1 + (S/6) * SR - ((K-3)/24) * SR^2)
    # where K is raw kurtosis. So (K-3) is excess kurtosis.
    # stats.kurtosis returns excess kurtosis.
    excess_k = stats.kurtosis(returns)
    
    # Adjustment
    adjustment = 1 + (s / 6.0) * sr - (excess_k / 24.0) * (sr ** 2)
    
    asr = sr * adjustment
    
    # Annualize? Usually Sharpe is annualized. 
    # But since we are comparing positions on the same timeframe, raw ASR is fine.
    # The prompt implies maximizing "expected adjusted Sharpe ratio" of next period returns,
    # or the simulated portfolio return series.
    # If the simulation is over N samples (drawn from distribution), it represents the distribution of outcomes.
    # So we calculate ASR of this distribution.
    
    return asr
