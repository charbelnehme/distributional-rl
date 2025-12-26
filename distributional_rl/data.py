import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=1000):
    """
    Generates synthetic returns and features for testing.
    
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target excess returns.
    """
    np.random.seed(42)
    
    # Generate random features
    momentum = np.random.normal(0, 1, n_samples)
    value = np.random.normal(0, 1, n_samples)
    volatility = np.abs(np.random.normal(0.02, 0.01, n_samples))
    
    X = pd.DataFrame({
        'momentum': momentum,
        'value': value,
        'volatility': volatility
    })
    
    # Generate target returns dependent on features
    # Returns = 0.01 * momentum + 0.005 * value + noise * volatility
    # Noise is Student-t to make it fat-tailed
    noise = np.random.standard_t(df=5, size=n_samples)
    
    y = 0.005 * momentum + 0.002 * value + noise * volatility
    
    return X, pd.Series(y, name='excess_return')
