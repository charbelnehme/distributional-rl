import optuna
import numpy as np
import pandas as pd
from distributional_rl.data import generate_synthetic_data
from distributional_rl.strategy import DistributionalStrategy
from distributional_rl.metrics import adjusted_sharpe_ratio

def objective(trial):
    # Generate data
    X, y = generate_synthetic_data(n_samples=500)
    
    # Time-series split (simple train/val split for demo)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Hyperparameters
    dist_name = trial.suggest_categorical('dist_name', ['Normal', 'StudentT'])
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    
    # Model Setup
    model_params = {
        'dist_name': dist_name,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }
    
    simulation_params = {
        'grid_points': np.linspace(0, 2, 11), # Coarse grid for speed
        'n_samples': 100 # Low samples for speed
    }
    
    strategy = DistributionalStrategy(model_params=model_params, simulation_params=simulation_params)
    
    try:
        strategy.fit(X_train, y_train)
        
        # Predict positions on validation set
        positions = strategy.predict_positions(X_val)
        
        # Calculate validation returns
        val_returns = positions * y_val.values
        
        # Calculate Adjusted Sharpe Ratio
        score = adjusted_sharpe_ratio(val_returns)
        
    except Exception as e:
        # Handle instability or errors gracefully
        print(f"Trial failed: {e}")
        return -1e9
        
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
