"""
Advanced feature engineering for baseball projections.

Incorporates statcast data, batted ball profiles, plate discipline metrics, and contextual factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

class FeatureEngineer:
  """
  Create advanced features for machine learning projections.

  Features include:
  - Statcast metrics (exit velo, launch angle, barrel rate)
  - Plate discipline (O-Swing%, Z-Contact%, SwStr%)
  - Batted ball distribution (GB%, LD%, FB%)
  - Quality of contact indicies
  - Historical trends
  - Contextual adjustments
  """

  def __init__(self):
    """Initialize feature engineer"""
    self.feature_definitions = self._define_features()

  def _define_features(self) -> Dict:
    """Define all feature types"""
    return {
      'power_features': [
        'exit_velo', 'max_exit_velo', 'barrel_rate', 'hard_hit_rate',
        'avg_launch_angle', 'fb_rate', 'hr_fb_rate', 'iso', 'slg'
      ],
      'contact_features': [
        'z_contact_rate', 'contact_rate', 'swstr_rate', 'avg',
        'babip', 'ld_rate', 'k_rate'
      ],
      'discipline_features': [
        'o_swing_rate', 'z_swing_rate', 'swing_rate', 'bb_rate',
        'chase_rate', 'first_pitch_strike_rate'
      ],
      'speed_features': [
        'sprint_speed', 'sb_rate', 'sb_success_rate', 'triples_rate'
      ],
      'batted_ball_features': [
        'gb_rate', 'fb_rate', 'ld_rate', 'iffb_rate', 'pull_rate',
        'cent_rate', 'oppo_rate'
      ],
      'trend_features': [
        'hr_growth', 'avg_growth', 'k_rate_change',
        'exit_velo_trend', 'consistency_score'
      ]
    }

  def calculate_power_score(self, df: pd.DataFrame) -> pd.Series:
    """
    Calculate composite power score.

    Combines exit velocity, barrel rate, HR/FB, and fly ball rate
    into a single power metric.

    Args:
      df: DataFrame with relevant columns

    Returns:
      Power score (z-score normalized)
    """
    power_components = []

    # Exit velocity (if available from statcast)
    if 'EV' in df.columns or 'exit_velocity' in df.columns:
      ev_col = 'EV' if 'EV' in df.columns else 'exit_velocity'
      ev_z = (df[ev_col] - df[ev_col].mean()) / df[ev_col].std()
      power_components.append(0.30 * ev_z)

    # Barrel rate
    if 'Barrel%' in df.columns or 'barrel_rate' in df.columns:
      barrel_col = 'Barrel%' if 'Barrel%' in df.columns else 'barrel_rate'
      barrel_z = (df[barrel_col] - df[barrel_col].mean()) / df[barrel_col].std()
      power_components.append(0.25 * barrel_z)

    # HR/FB rate
    if 'HR/FB' in df.columns or 'hr_fb_rate' in df.columns:
      hrfb_col = 'HR/FB' if 'HR/FB' in df.columns else 'hr_fb_rate'
      hrfb_z = (df[hrfb_col] - df[hrfb_col].mean()) / df[hrfb_col].std()
      power_components.append(0.25 * hrfb_z)

    # ISO
    if 'ISO' in df.columns:
      iso_z = (df['ISO'] - df['ISO'].mean()) / df['ISO'].std()
      power_components.append(0.20 * iso_z)

    if len(power_components) == 0:
      return pd.Series(0, index=df.index)

    return sum(power_components)

  def calculate_contact_quality(self, df: pd.DataFrame) -> pd.Series:
    """
    Calculate contact quality index.

    Args:
      df: DataFrame with contact metrics

    Returns:
      Contact quality score
    """
    contact_components = []

    # Zone contact rate
    if 'Z-Contact%' in df.columns or 'z_contact_rate' in df.columns:
      zcon_col = 'Z-Contact%' if 'Z-Contact%' in df.columns else 'z_contact_rate'
      zcon_z = (df[zcon_col] - df[zcon_col].mean()) / df[zcon_col].std()
      contact_components.append(0.35 * zcon_z)

    # Hard hit rate
    if 'HardHit%' in df.columns or 'hard_hit_rate' in df.columns:
      hh_col = 'HardHit%' if 'HardHit%' in df.columns else 'hard_hit_rate'
      hh_z = (df[hh_col] - df[hh_col].mean()) / df[hh_col].std()
      contact_components.append(0.30 * zcon_z)

    # Swinging strike rate (inverse - lower is better)
    if 'SwStr%' in df.columns or 'swstr_rate' in df.columns:
      swstr_col = 'SwStr%' if 'SwStr%' in df.columns else 'swstr_rate'
      swstr_z = (df[swstr_col] - df[swstr_col].mean()) / df[swstr_col].std()
      contact_components.append(-0.25 * swstr_z)

    # Line drive rate
    if 'LD%' in df.columns or 'ld_rate' in df.columns:
      ld_col = 'LD%' if 'LD%' in df.columns else 'ld_rate'
      ld_z = (df[ld_col] - df[ld_col].mean()) / df[ld_col].std()
      contact_components.append(0.10 * ld_z)

    if len(contact_components) == 0:
      return pd.Series(0, index=df.index)

    return sum(contact_components)

  def calculate_plate_discipline_score(self, df: pd.DataFrame) -> pd.Series:
    """
    Calculate plate discipline index

    Args:
      df: DataFrame with discipline metrics

    Returns:
      Discipline score
    """
    discipline_components = []

    # O-Swing% (lower is better)
    if 'O-Swing%' in df.columns or 'o_swing_rate' in df.columns:
      oswing_col = 'O-Swing%' if 'O-Swing%' in df.columns else 'o_swing_rate'
      oswing_z = (df[oswing_col] - df[oswing_col].mean()) / df[oswing_col].std()
      discipline_components.append(-0.40 * oswing_z)

    # Walk rate (higher is better)
    if 'BB%' in df.columns:
      bb_z = (df['BB%'] - df['BB%'].mean()) / df['BB%'].std()
      discipline_components.append(0.35 * bb_z)

    # Zone swing rate
    if 'Z-Swing%' in df.columns or 'z_swing_rate' in df.columns:
      zswing_col = 'Z-Swing%' if 'Z-Swing%' in df.columns else 'z_swing_rate'
      zswing_z = (df[zswing_col] - df[zswing_col].mean()) / df[zswing_col].std()
      discipline_components.append(0.25 * zswing_z)

    if len(discipline_components) == 0:
      return pd.Series(0, index=df.index)

    return sum(discipline_components)

  def calculate_speed_score(self, df: pd.DataFrame) -> pd.Series:
    """
    Calculate speed/athleticism score.

    Args:
      df: DataFrame with speed metrics

    Returns:
      Speed scores
    """
    speed_components = []

    # Sprint speed (if available from statcast)
    if 'Sprint Speed' in df.columns or 'sprint_speed' in df.columns:
      sprint_col = 'Sprint Speed' if 'Sprint Speed' in df.columns else 'sprint_speed'
      sprint_z = (df[sprint_col] - df[sprint_col].mean()) / df[sprint_col].std()
      speed_components.append(0.50 * sprint_z)

    # Stolen base rate
    if 'SB_rate' in df.columns:
      sb_z = (df['SB_rate'] - df['SB_rate'].mean()) / df['SB_rate'].std()
      speed_components.append(0.30 * sb_z)

    # Triples rate
    if '3B' in df.columns and 'AB' in df.columns:
      triples_rate = df['3B'] / df['AB']
      triples_z = (triples_rate - triples_rate.mean()) / triples_rate.std()
      speed_components.append(0.20 * triples_z)

    if len(speed_components) == 0:
      return pd.Series(0, index=df.index)

    return sum(speed_components)

  def calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance trends over time for each player.

    Args:
      df: DataFrame with multiple seasons per player

    Returns:
      DataFrame with trend features added
    """
    df = df.copy()
    trend_features = []

    for player_id in df['player_id'].unique():
      player_data = df[df['player_id'] == player_id].sort_values('Season')

      if len(player_data) < 2:
        for col in ['HR_growth', 'AVG_growth', 'K_rate_change', 'consistency_score']:
          player_data[col] = 0
        trend_features.append(player_data)
        continue

      # HR growth (comparing recent to older)
      if 'HR' in player_data.columns and 'PA' in player_data.columns:
        recent_hr_rate = player_data.iloc[-1]['HR'] / player_data.iloc[-1]['PA']
        old_hr_rate = player_data.iloc[0]['HR'] / player_data.iloc[0]['PA']
        player_data['HR_growth'] = recent_hr_rate - old_hr_rate
      else:
        player_data['HR_growth'] = 0

      # AVG growth
      if 'AVG' in player_data.columns:
        player_data['AVG_growth'] = (
          player_data.iloc[-1]['AVG'] - player_data.iloc[0]['AVG']
        )
      else:
        player_data['AVG_growth'] = 0

      # K rate change
      if 'K%' in player_data.columns:
        player_data['K_rate_change'] = (
          player_data.iloc[-1]['K%'] - player_data.iloc[0]['K%']
        )
      else:
        player_data['K_rate_change'] = 0

      # Consistency (inverse of standard deviation of performance)
      if 'wOBA' in player_data.columns and len(player_data) > 3:
        consistency = 1 / (player_data['wOBA'].std() + 0.01)
        player_data['consistency_score'] = consistency
      else:
        player_data['consistency_score'] = 0

      trend_features.append(player_data)

    return pd.concat(trend_features, ignore_index=True)

  def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features (combinations of existing factors).

    Args:
      df: DataFrame with base features

    Returns:
      DataFrame with interaction features added
    """
    df = df.copy()

    # Power * Contact = Quality hitter
    if 'ISO' in df.columns and 'AVG' in df.columns:
      df['power_contact_interaction'] = df['ISO'] * df['AVG']

    # Patience * Power = Productive hitter
    if 'BB%' in df.columns and 'SLG' in df.columns:
      df['patience_power'] = df['BB%'] * df['SLG']

    # Young * improving = breakout candidate
    if 'Age' in df.columns and 'HR_growth' in df.columns:
      df['young_improver'] = (30 - df['Age']) * df['HR_growth']
      df['young_improver'] = df['young_improver'].clip(lower=0)

    # Contact * speed = Table setter
    if 'AVG' in df.columns and 'SB_rate' in df.columns:
      df['contact_speed'] = df['AVG'] * df['SB_rate']

    return df

  def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering.

    Args:
      df: Raw batting data

    Returns:
      DataFrame with engineered features
    """
    print("Engineering features...")

    df = df.copy()

    # Calculate composite scores
    df['power_score'] = self.calculate_power_score(df)
    df['contact_quality'] = self.calculate_contact_quality(df)
    df['discipline_score'] = self.calculate_plate_discipline_score(df)
    df['speed_score'] = self.calculate_speed_score(df)

    # Calculate (if multiple seasons per player)
    if 'player_id' in df.columns:
      df = self.calculate_trend_features(df)

    # Create interactions
    df = self.create_interaction_features(df)

    # Fill NaN with 0 for new features
    new_features = [
      'power_score', 'contact_quality', 'discipline_score', 'speed_score',
      'HR_growth', 'AVG_growth', 'K_rate_change', 'consistency_score',
      'power_contact_interaction', 'patience_power', 'young_improver',
      'contact_speed'
    ]

    for feature in new_features:
      if feature in df.columns:
        df[feature] = df[feature].fillna(0)

    print(f"Added {len(new_features)} engineered features.")

    return df

def get_feature_importance_groups() -> Dict[str, List[str]]:
  """
  Return feature groups for analysis and selection.

  Returns:
    Dict mapping group names to feature lists
  """
  return {
    'traditional': [
      'AVG', 'OBP', 'SLG', 'HR', 'RBI', 'R', 'SB', 'BB%', 'K%', 'ISO', 'wOBA'
    ],
    'statcast': [
      'EV', 'LA', 'Barrel%', 'HArdHit%', 'Sprint Speed'
    ],
    'batted_ball': [
      'GB%', 'FB%', 'LD%', 'HR/FB', 'Pull%', 'Cent%', 'Oppo%'
    ],
    'discipline': [
      'O-Swing%', 'Z-Swing%', 'Z-Contact%', 'SwStr%', 'Chase%'
    ],
    'composite': [
      'power_score', 'contact_quality', 'discipline_score', 'speed_score',
    ],
    'trends': [
      'HR_growth', 'AVG_growth', 'K_rate_change', 'consistency_score'
    ],
    'interactions': [
      'power_contact_interaction', 'patience_power', 'young_improver', 'contact_speed'
    ],
    'context': [
      'Age', 'PA'
    ]
  }

if __name__ == "__main__":


  from src.simple_projections.data_loader import load_player_data

  print("Loading data...")
  data = load_player_data([2024, 2025], min_pa=300)

  print("\nEngineering features...")
  engineer = FeatureEngineer()
  enhanced_data = engineer.engineer_features(data)

  print(f"\nOriginal columns: {len(data.columns)}")
  print(f"Enhanced columns: {len(enhanced_data.columns)}")

  print("\nNew features:")
  new_cols = set(enhanced_data.columns) - set(data.columns)
  print(list(new_cols))

  print("\nSample enhanced data:")
  display_cols = ['Name', 'Age', 'AVG', 'HR', 'power_score',
                  'contact_quality', 'discipline_score', 'speed_score']
  available = [c for c in display_cols if c in enhanced_data.columns]
  print(enhanced_data[available].head(10))