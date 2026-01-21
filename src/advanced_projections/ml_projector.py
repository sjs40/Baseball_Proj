"""
Machine Learning Baseball Projection System

Uses Random Forest, XGBoost, and ensemble methods for advanced projections
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
  import xgboost as xgb
  HAS_XGBOOST = True
except ImportError:
  HAS_XGBOOST = False
  warnings.warn("XGBoost not available, will use Random Forest only.")

from .feature_engineering import FeatureEngineer
from .park_factors import ParkFactors
from ..simple_projections.aging_curves import AgingCurve

warnings.filterwarnings("ignore")

class MLProjector:
  """
  Advanced ML-based projection system

  Uses ensemble of models with engineered features, park adjustments and uncertainty quantification
  """

  def __init__(self, use_xgboost: bool = True, n_estimators: int = 300, random_state: int = 42):
    """
    Initialize MLProjector

    Args:
      use_xgboose: Whether to use XGBoost
      n_estimators: Number of trees for ensemble models
      random_state: Random seed for reproducibility
    """
    self.use_xgboost = use_xgboost and HAS_XGBOOST
    self.n_estimators = n_estimators
    self.random_state = random_state

    # Components
    self.feature_engineer = FeatureEngineer()
    self.park_factors = ParkFactors()
    self.aging_curves = AgingCurve()

    # Model (one per target variable)
    self.models = {}
    self.scalers = {}
    self.feature_names = {}

    # Training metadata
    self.training_metrics = {}

  def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling

    Args:
      df: raw batting data

    Returns:
      DataFrame with engineered features
    """
    # Engineer features
    df_enhanced = self.feature_engineer.engineer_features(df)

    return df_enhanced

  def select_features_for_target(self, df: pd.DataFrame, target: str) -> List[str]:
    """
    Select relevant features for specific target variable.

    Args:
      df: DataFrame with all features
      target: Target variable to predict

    Returns:
      List of feature names
    """
    # Base features always included
    base_features = ['Age', 'PA']

    # Target-specific features
    target_feature_map = {
      'HR': [
        'power_score', 'ISO', 'FB%', 'HR/FB', 'Barrel%', 'EV',
        'hard_hit_rate', 'HR_growth', 'power_contact_interaction',
        'AVG', 'SLG'
      ],
      'AVG': [
        'contact_quality', 'Z-Contact%', 'SwStr%', 'K%', 'BABIP',
        'LD%', 'consistency_score', 'AVG_growth'
      ],
      'OBP': [
        'discipline_score', 'BB%', 'O-Swing%', 'AVG', 'patience_power',
        'Z-Contact%'
      ],
      'SLG': [
        'power_score', 'ISO', 'HR', 'contact_quality', 'Barrel%',
        'power_contact_interaction'
      ],
      'wOBA': [
        'power_score', 'contact_quality', 'discipline_score',
        'AVG', 'OBP', 'SLG', 'ISO', 'BB%', 'K%'
      ],
      'SB': [
        'speed_score', 'Sprint Speed', 'SB_rate', 'Age',
        'contact_speed', 'AVG'
      ]
    }

    # Get target-specific features
    specific_features = target_feature_map.get(target, [])

    # Combine with base features
    all_features = base_features + specific_features

    # Filter to features that exist in df
    available_features = [f for f in all_features if f in df.columns]

    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in available_features:
      if f not in seen:
        seen.add(f)
        unique_features.append(f)

    return unique_features

  def train_model_for_target(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Tuple[object, Dict]:
    """
    Train model for specific target variable.

    Args:
      X: Feature DataFrame
      y: Target variable
      target_name: Name of target variable

    Returns:
      Tuple of (trained_model, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model
    if self.use_xgboost:
      model = xgb.XGBRegressor(
        n_estimators=self.n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=self.random_state,
        n_jobs=-1
      )
    else:
      model = RandomForestRegressor(
        n_estimators=self.n_estimators,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=self.random_state,
        n_jobs=-1
      )

    # Train
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    metrics = {
      'train_mae': mean_absolute_error(y_train, y_pred_train),
      'test_mae': mean_absolute_error(y_test, y_pred_test),
      'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
      'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
      'train_r2': r2_score(y_train, y_pred_train),
      'test_r2': r2_score(y_test, y_pred_test),
      'n_train': len(X_train),
      'n_test': len(X_test)
    }

    # Cross-validation
    cv_scores = cross_val_score(
      model, X_train_scaled, y_train,
      cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    metrics['cv_mae'] = -cv_scores.mean()
    metrics['cv_mae_std'] = cv_scores.std()

    print(f"\n{target_name} Model Performance:")
    print(f"  Test MAE: {metrics['test_mae']:.3f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"  Test R2: {metrics['test_r2']:.3f}")
    print(f"  CV MAE: {metrics['cv_mae']:.3f} +/- {metrics['cv_mae_std']:.3f}")

    return model, scaler, metrics

  def train(self, df: pd.DataFrame, targets: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Train models for all target variables.

    Args:
      df: Historical batting data with multiple seasons
      targets: List of stats to predict (None = default set)

    Returns:
      Dict of training metrics per target
    """
    print("Preparing data for training...")

    # Prepare features
    df_prepared = self.prepare_features(df)

    # Default targets if not specified
    if targets is None:
      targets = ['HR', 'AVG', 'OBP', 'SLG', 'wOBA', 'SB']

    # Filter targets to those available
    targets = [t for t in targets if t in df_prepared.columns]

    print(f"\nTraining models for: {targets}")

    all_metrics = {}

    for target in targets:
      print(f"\n{'='*60}")
      print(f"Training {target} model...")
      print('='*60)

      # Select features for this target
      feature_cols = self.select_features_for_target(df_prepared, target)

      print(f"Using {len(feature_cols)} features:")
      print(f"  {feature_cols}")

      # Prepare X and y
      df_target = df_prepared[feature_cols + [target]].dropna()
      X = df_target[feature_cols]
      y = df_target[target]

      # Train model
      model, scaler, metrics = self.train_model_for_target(X, y, target)

      # Store model and metadata
      self.models[target] = model
      self.scalers[target] = scaler
      self.feature_names[target] = feature_cols
      all_metrics[target] = metrics

    self.training_metrics = all_metrics

    print(f"\n{'='*60}")
    print("Training complete!")
    print('='*60)

    return all_metrics

  def predict_player(self, player_data: pd.DataFrame, target_year: int) -> pd.Series:
    """
    Generate projection for a single player.

    Args:
      player_data: Historical stats for player
      target_year: Year to project

    Returns:
      Series with projections
    """
    if len(player_data) == 0:
      return pd.Series()

    # Get player info
    player_name = player_data.iloc[-1]['Name']
    last_season = player_data.iloc[-1]['Season']
    last_age = player_data.iloc[-1]['Age']

    # Calculate projected age
    years_forward = target_year - last_season
    projected_age = last_age + years_forward

    # Prepare features from most recent season
    recent_data = player_data.iloc[[-1]].copy()
    recent_data = self.feature_engineer.engineer_features(recent_data)

    # Create projection dict
    projection = {
      'Name': player_name,
      'Age': projected_age
    }

    # Predict each target
    for target, model in self.models.items():
      # Get features for this target
      feature_cols = self.feature_names[target]

      # Check if all features available
      available_features = [f for f in feature_cols if f in recent_data.columns]

      if len(available_features) < len(feature_cols) * 0.7:
        continue

      # Prepare feature vector
      X = recent_data[feature_cols].fillna(0)

      # Scale features
      scaler = self.scalers[target]
      X_scaled = scaler.transform(X)

      # Predict
      pred = model.predict(X_scaled)[0]

      projection[target] = pred

    # Estimate playing time (use simple model)
    if 'PA' in player_data.columns:
      recent_pa = player_data.tail(3)['PA'].mean()

      # Age adjustment
      if projected_age > 32:
        age_factor = 1.0 - (projected_age - 32) * 0.03
        age_factor = max(age_factor, 0.6)
      elif projected_age < 25:
        age_factor = 0.95 + (25 - projected_age) * 0.02
        age_factor = min(age_factor, 1.05)
      else:
        age_factor = 1.0

      projected_pa = min(recent_pa * age_factor, 700)
      projection['PA'] = int(projected_pa)

    return pd.Series(projection)

  def predict(self, df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Generate projections for all players.

    Args:
      df: Historical batting data
      target_year: Year to project

    Returns:
      DataFrame with projections
    """
    if not self.models:
      raise ValueError("Models not trained. Call train() first.")

    print(f"Generating ML projections for {target_year}...")

    # Get unique players
    player_ids = df['player_id'].unique()

    projections = []

    for player_id in player_ids:
      player_data = df[df['player_id'] == player_id].sort_values('Season')

      # Only project if recent data
      last_season = player_data.iloc[-1]['Season']
      if target_year - last_season > 2:
        continue

      projection = self.predict_player(player_data, target_year)

      if len(projection) > 0:
        projections.append(projection)

    projections_df = pd.DataFrame(projections)

    print(f"Generated {len(projections_df)} projections.")

    return projections_df

  def get_feature_importance(self, target: str, top_n: int = 15) -> pd.DataFrame:
    """
    Get feature importane for a target model.

    Args:
      target: Target variable
      top_n: Number of top features to return

    Returns:
      DataFrame with feature importance
    """
    if target not in self.models:
      raise ValueError(f"Model not trained for {target}.")

    model = self.models[target]
    feature_names = self.feature_names[target]

    if hasattr(model, 'feature_importances_'):
      importances = model.feature_importances_
    else:
      return pd.DataFrame()

    # Create DataFrame
    importance_df = pd.DataFrame({
      'Feature': feature_names,
      'Importance': importances
    })

    importance_df = importance_df.sort_values('Importance', ascending=False)

    return importance_df.head(top_n)
