"""
Example: Advanced Baseball Projections with Machine Learning

This demonstrates how to create advanced projections using
ML models, feature engineering, and ensemble methods
"""

import sys
sys.path.append('..')

from src.simple_projections.data_loader import load_player_data
from src.simple_projections.projector import SimpleProjector
from src.advanced_projections.ml_projector import MLProjector
from src.advanced_projections.feature_engineering import FeatureEngineer
from src.advanced_projections.park_factors import ParkFactors
from src.utils.metrics import ProjectionEvaluator

import pandas as pd
import numpy as np

def main():
  print("="*80)
  print("ADVANCED BASEBALL PROJECTION SYSTEM - EXAMPLE")
  print("="*80)

  # Step 1: load training data
  print("\n[Step 1] Loading training data...")
  print("Years: 2022, 2023, 2024, 2025")

  training_data = load_player_data([2022, 2023, 2024, 2025], min_pa=250)

  print(f"Loaded {len(training_data)} player-seasons for training")

  # Step 2: Initialize and train ML projector
  print("\n[Step 2] Training ML Projection Models...")
  print("This may take several minutes...")

  ml_projector = MLProjector(
    use_xgboost=True,
    n_estimators=300,
    random_state=42
  )

  # Train models
  metrics = ml_projector.train(training_data)

  print("\n\nTraining Complete!")
  print("\nModel Performance Summary:")
  print("-"*60)

  for target, metric_dict in metrics.items():
    print(f"\n{target}:")
    print(f"  Test MAE: {metric_dict['test_mae']:.3f}")
    print(f"  Test R^2: {metric_dict['test_r2']:.3f}")
    print(f"  CV MAE: {metric_dict['cv_mae']:.3f}")

  # Step 3: Generate projections for 2026
  print("\n\n[Step 3] Generating Projections for 2026...")

  ml_projections = ml_projector.predict(training_data, 2026)

  print(f"Created {len(ml_projections)} ML projections")

  # Step 4: Compare with simple projection
  simple_projector = SimpleProjector()
  simple_projections = simple_projector.project(training_data, 2026)

  print(f"Created {len(simple_projections)} simple projections")

  # Step 5: Feature importance analysis
  print("\n\n[Step 5] Analyzing Feature Importance...")

  targets_to_analyze = ['HR', 'AVG', 'wOBA']

  for target in targets_to_analyze:
    if target in ml_projections.columns:
      print(f"\n{target} - Top 10 Features:")
      print("-"*60)

      importance_df = ml_projector.get_feature_importance(target, top_n=10)
      print(importance_df.to_string(index=False))

      # TODO Plot feature importance

  # Step 6: Display top projections
  print("\n\n[Step 6] Top 15 Players - ML Projections:")
  print("="*80)

  display_cols = ['Name', 'Age', 'PA', 'AVG', 'HR', 'OBP', 'SLG', 'wOBA', 'SB']
  available_cols = [col for col in display_cols if col in ml_projections.columns]

  # Sort by wOBA if available, else by first numeric column
  if 'wOBA' in ml_projections.columns:
    top_15 = ml_projections.nlargest(15, 'wOBA')[available_cols]
  else:
    numeric_cols = ml_projections.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
      top_15 = ml_projections.nlargest(15, numeric_cols[0])[available_cols]
    else:
      top_15 = ml_projections[available_cols].head(15)

  print(top_15.to_string(index=False))

  # Step 7: Park factor analysis
  print("\n\n[Step 7] Park Factor Analysis...")

  park_factors = ParkFactors()

  print("\nMost Hitter-Friendly Park (Overall):")
  rankings = park_factors.get_park_rankings('overall')
  print(rankings.head(5).to_string(index=False))

  # Example park adjustment
  print("\n\nExample: Player moving from Coors to Oracle Park")
  coors_hr = 35
  adjusted_hr = park_factors.adjust_stat(coors_hr, 'COL', 'SF', 'HR')
  print(f"  Original projection at Coors: {coors_hr} HR")
  print(f"  Adjusted for Oracle Park: {adjusted_hr:.1f} HR")
  print(f"  Difference: {coors_hr - adjusted_hr:.1f} HR")

  # Step 8: Ensemble projections
  print("\n\n[Step 8] Creating Ensemble Projections...")
  print("Combining ML and Simple projections...")

  # Merge projections
  merged = ml_projections.merge(
    simple_projections[['Name', 'HR', 'AVG', 'OBP', 'SLG']],
    on='Name',
    suffixes=('_ml', '_simple'),
    how='inner'
  )

  # Create ensemble (weighted average)
  for stat in ['HR', 'AVG', 'OBP', 'SLG']:
    if f'{stat}_ml' in merged.columns and f'{stat}_simple' in merged.columns:
      merged[f'{stat}_ensemble'] = (
        0.6 * merged[f'{stat}_ml'] + 0.4 * merged[f'{stat}_simple']
      )

  print(f"Created ensemble projections for {len(merged)} players")

  print("\nEnsemble Projection Examples:")
  print("-"*80)

  ensemble_cols = ['Name', 'HR_ensemble', 'AVG_ensemble', 'OBP_ensemble', 'SLG_ensemble']
  available_ensemble = [c for c in ensemble_cols if c in merged.columns]

  if 'HR_ensemble' in merged.columns:
    top_ensemble = merged.nlargest(15, 'HR_ensemble')[available_ensemble]
    print(top_ensemble.to_string(index=False))

  # Step 9: Save results
  print("\n\n[Step 9] Saving results...")

  ml_projections.to_csv('ml_projections_2026.csv', index=False)
  print("Saved ML projections to: ml_projections_2026.csv")

  if len(merged) > 0:
    merged.to_csv('ensemble_projections_2026.csv', index=False)
    print("Saved ensemble projections to: ensemble_projections_2026.csv")

  # Step 10: Summary
  print("\n\n"+ "="*80)
  print("ADVANCED PROJECTION SUMMARY")
  print("="*80)

  print(f"\nTotal ML Projections: {len(ml_projections)}")
  print(f"Total Simple Projections: {len(simple_projections)}")
  print(f"Total Ensemble Projections: {len(merged)}")



if __name__ == "__main__":
  main()

