"""
Evaluation metrics for baseball projections.

Compare projections against actual results to assess accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

class ProjectionEvaluator:
  """Evaluate projections against actual results"""

  def __init__(self):
    self.results = {}

  def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean absolute error"""
    return mean_absolute_error(actual, predicted)

  def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate root mean squared error"""
    return np.sqrt(mean_squared_error(actual, predicted))

  def calculate_r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R^2"""
    return r2_score(actual, predicted)

  def calculate_correlation(self, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient"""
    return np.corrcoef(actual, predicted)[0, 1]

  def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean absolute percentage error"""
    mask = actual != 0
    if mask.sum() == 0:
      return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

  def evaluate_stat(self, projections: pd.DataFrame, actuals: pd.DataFrame, stat: str, merge_on: str = 'Name') -> Dict[str, float]:
    """
    Evaluate projections for specific stat.

    Args:
      projections: DataFrame with projected stats
      actuals: DataFrame with actual stats
      stat: Stat to evaluate
      merge_on: Column to merge on

    Returns:
      Dict of evaluation metrics
    """
    # Merge projections with actuals
    merged = projections.merge(
      actuals[[merge_on, stat]],
      on=merge_on,
      suffixes=('_proj', '_actual')
    )

    # Get columns
    proj_col = f'{stat}_proj' if f'{stat}_proj' in merged.columns else stat
    actual_col = f'{stat}_actual'

    # Filter complete cases
    merged = merged[[proj_col, actual_col]].dropna()

    if len(merged) == 0:
      return {}

    projected = merged[proj_col].values
    actual = merged[actual_col].values

    # Calculate metrics
    metrics = {
      'stat': stat,
      'n': len(merged),
      'mae': self.calculate_mae(actual, projected),
      'rmse': self.calculate_rmse(actual, projected),
      'r2': self.calculate_r2(actual, projected),
      'correlation': self.calculate_correlation(actual, projected),
      'mape': self.calculate_mape(actual, projected),
      'mean_actual': actual.mean(),
      'mean_projected': projected.mean(),
      'std_actual': actual.std(),
      'std_projected': projected.std()
    }

    return metrics

  def evaluate_multiple_stats(self, projections: pd.DataFrame, actuals: pd.DataFrame, stats: List[str], merge_on: str = 'Name') -> pd.DataFrame:
    """
    Evaluate multiple stats at once.

    Args:
      projections: Projected stats
      actuals: Actual stats
      stats: List of stats to evaluate
      merge_on: Column to merge on

    Returns:
      DataFrame with metrics for each stat
    """
    results = []

    for stat in stats:
      if stat not in projections.columns or stat not in actuals.columns:
        continue

      metrics = self.evaluate_stat(projections, actuals, stat, merge_on)

      if metrics:
        results.append(metrics)

    if not results:
      return pd.DataFrame()

    return pd.DataFrame(results)

  def print_evaluation_summary(self, metrics_df: pd.DataFrame):
    """Print formatted evaluation summary"""
    print("\n" + "="*80)
    print("PROJECTION EVALUATION SUMMARY")
    print("="*80)

    for _, row in metrics_df.iterrows():
      stat = row['stat']
      print(f"\n{stat}:")
      print(f"  Sample Size: {int(row['n'])}")
      print(f"  MAE: {row['mae']:.3f}")
      print(f"  RMSE: {row['rmse']:.3f}")
      print(f"  R^2: {row['r2']:.3f}")
      print(f"  Correlation: {row['correlation']:.3f}")
      if not np.isnan(row.get('mape', np.nan)):
        print(f"  MAPE: {row['mape']:.1f}%")
      print(f"  Mean Actual: {row['mean_actual']:.3f}")
      print(f"  Mean Projected: {row['mean_projected']:.3f}")

  def compare_system(self, system1_projs: pd.DataFrame, system2_projs: pd.DataFrame, actuals: pd.DataFrame, stats: List[str], system1_name: str = 'System 1', system2_name: str = 'System 2') -> pd.DataFrame:
    """
    Compare two projection systems

    Args:
      system1_projs: Projections from system 1
      system2_projs: Projections from system 2
      actuals: Actual results
      stats: Stats to compare
      system1_name: Name of system 1
      system2_name: Name of system 2

    Returns:
      Comparison DataFrame
    """
    # Evaluate both systems
    metrics1 = self.evaluate_multiple_stats(system1_projs, actuals, stats)
    metrics2 = self.evaluate_multiple_stats(system2_projs, actuals, stats)

    if len(metrics1) == 0 or len(metrics2) == 0:
      return pd.DataFrame()

    # Merge metrics
    comparison = metrics1[['stat', 'mae', 'rmse', 'r2']].merge(
      metrics2[['stat', 'mae', 'rmse', 'r2']],
      on='stat',
      suffixes=(f'_{system1_name}', f'_{system2_name}')
    )

    # Calculate improvements
    comparison['mae_improvement'] = (
      (comparison[f'mae_{system2_name}'] - comparison[f'mae_{system1_name}']) / comparison[f'mae_{system1_name}'] * 100
    )

    comparison['r2_improvement'] = (
      comparison[f'r2_{system2_name}'] - comparison[f'r2_{system1_name}']
    )

    return comparison

  def identify_outliers(self, projections: pd.DataFrame, actuals: pd.DataFrame, stat: str, threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify players with large projection errors

    Args:
      projections: Projectioned stats
      actuals: Actual stats
      stat: Stat to check
      threshold: Z-score threshold for outliers

    Returns:
      DataFrame with outlier players
    """
    # Merge
    merged = projections.merge(
      actuals[['Name', stat]],
      on='Name',
      suffixes=('_proj', '_actual')
    )

    proj_col = f'{stat}_proj' if f'{stat}_proj' in merged.columns else stat
    actual_col = f'{stat}_actual'

    merged = merged[['Name', proj_col, actual_col]].dropna()

    # Claculate errors
    merged['error'] = merged[actual_col] - merged[proj_col]
    merged['abs_error'] = np.abs(merged['error'])

    # Z-score of errors
    error_mean = merged['error'].mean()
    error_std = merged['error'].std()
    merged['error_zscore'] = (merged['error'] - error_mean) / error_std

    # Identify outliers
    outliers = merged[np.abs(merged['error_zscore']) > threshold]
    outliers = outliers.sort_values('abs_error', ascending=False)

    return outliers[['Name', proj_col, actual_col, 'error', 'error_zscore']]

def calculate_percentile_accuracy(projections: pd.DataFrame, actuals: pd.DataFrame, stat: str, percentiles: List[int] = [10,50,90]) -> Dict:
  """
  Calculate accuracy at different percentiles

  Useful for evaluating confidence intervals.

  Args:
    projections: Must have columns like '{stat}_p10', '{stat}_p50', '{stat}_p90'
    actuals: Actual results
    stat: Stat to evaluate
    percentiles: Which percentiles to evaluate

  Returns:
    Dict with coverage statistics
  """
  results = {}

  # Merge data
  merged = projections.merge(actuals[['Name', stat]], on='Name')
  merged = merged.dropna()

  actual_values = merged[stat].values

  # Check each percentile
  for p in percentiles:
    p_col = f'{stat}_p{p}'

    if p_col not in merged.columns:
      continue

    projected_p = merged[p_col].values

    if p == 50:
      # Median - calculate MAE
      mae = mean_absolute_error(actual_values, projected_p)
      results[f'p{p}_mae'] = mae
    elif p < 50:
      # Lower bound - check coverage
      below = (actual_values < projected_p).mean()
      results[f'p{p}_coverage'] = below * 100
    else:
      # Upper bound - check coverage
      above = (actual_values > projected_p).mean()
      results[f'p{p}_coverage'] = above * 100

  return results

