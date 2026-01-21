"""Simple projection system using weighted averages"""

import pandas as pd
import numpy as np
from .aging_curves import AgingCurve

class SimpleProjector:
  """Simple projection system"""

  def __init__(self, weights=[5,4,3], regression_factor=0.20, min_pa=200):
    self.weights = weights
    self.regression_factor = regression_factor
    self.min_pa = min_pa
    self.aging_curve = AgingCurve()
    self.league_averages = None

  def calculate_league_averages(self, df):
    """Calculate league averages for regression"""
    total_pa = df['PA'].sum()
    averages = {}

    for stat in ['AVG', 'OBP', 'SLG', 'wOBA']:
      if stat in df.columns:
        averages[stat] = (df[stat] * df['PA']).sum() / total_pa

    if 'HR' in df.columns:
      averages['HR_rate'] = df['HR'].sum() / total_pa

    self.league_averages = averages
    return averages

  def calculate_weighted_average(self, player_data):
    """Calculate weighted average of recent seasons"""
    recent = player_data.tail(len(self.weights))

    if len(recent) == 0:
      return pd.Series()

    n_seasons = len(recent)
    season_weights = self.weights[-n_seasons:]
    normalized_weights = np.array(season_weights) / sum(season_weights)

    weighted_stats = {}
    for col in recent.columns:
      if col in ['Name', 'Season', 'Age', 'player_id', 'Team']:
        continue
      if recent[col].dtype in [np.float64, np.int64]:
        weighted_stats[col] = np.average(recent[col].values, weights=normalized_weights)

    return pd.Series(weighted_stats)

  def regress_to_mean(self, stats):
    """Regress extreme values toward league averages"""
    if self.league_averages is None:
      return stats

    regressed = stats.copy()
    for stat, league_avg in self.league_averages.items():
      if stat in regressed.index:
        player_value = regressed[stat]
        regressed[stat] = ((1 - self.regression_factor) * player_value + self.regression_factor * league_avg)

    return regressed

  def project_playing_time(self, player_data, projected_age):
    """Project plate appearances"""
    recent_pa = player_data.tail(len(self.weights))['PA']

    if len(recent_pa) == 0:
      return 500

    weights = self.weights[-len(recent_pa):]
    normalized = np.array(weights) / sum(weights)
    base_pa = np.average(recent_pa.values, weights=normalized)

    # Age adjustment
    if projected_age >= 32:
      age_factor = max(1.0 - (projected_age -32) * 0.03, 0.6)
    elif projected_age <= 25:
      age_factor = min(0.95 + (25 - projected_age) * 0.02, 1.05)
    else:
      age_factor = 1.0

    projected_pa = base_pa * age_factor
    return min(max(projected_pa, 200), 700)

  def project_player(self, player_data, target_year):
    """Create projection for a single player"""
    if len(player_data) == 0:
      return pd.Series()

    player_name = player_data.iloc[-1]['Name']
    last_season = player_data.iloc[-1]['Season']
    last_age = player_data.iloc[-1]['Age']

    years_forward = target_year - last_season
    projected_age = last_age + years_forward

    # Weighted average
    weighted_avg = self.calculate_weighted_average(player_data)
    if len(weighted_avg) == 0:
      return pd.Series()

    # Apply aging
    aged_stats = self.aging_curve.apply_aging_to_stats(
      weighted_avg,
      int(weighted_avg.get('Age', last_age)),
      projected_age
    )

    # Regress to mean
    regressed_stats = self.regress_to_mean(aged_stats)

    # Projected playing time
    projected_pa = self.project_playing_time(player_data, projected_age)

    # Build projection
    projection = pd.Series({
      'Name': player_name,
      'Age': projected_age,
      'PA': int(projected_pa)
    })

    # Add rate stats
    for stat in ['AVG', 'OBP', 'SLG', 'wOBA']:
      if stat in regressed_stats.index:
        projection[stat] = regressed_stats[stat]

    # Calculate counting stats
    if 'AVG' in projection.index:
      ab = projected_pa * 0.9
      projection['H'] = int(projection['AVG'] * ab)
      projection['AB'] = int(ab)

    if 'HR_rate' in regressed_stats.index:
      projection['HR'] = int(regressed_stats['HR_rate'] * projected_pa)

    return projection

  def project(self, df, target_year, players=None):
    """Create projections for multiple players"""
    self.calculate_league_averages(df)

    player_ids = (df[df['Name'].isin(players)]['player_id'].unique() if players else df['player_id'].unique())
    projections = []
    for player_id in player_ids:
      player_data = df[df['player_id'] == player_id].sort_values('Season')

      last_season = player_data.iloc[-1]['Season']
      if target_year - last_season > 2:
        continue

      projection = self.project_player(player_data, target_year)
      if len(projection) > 0:
        projections.append(projection)

    return pd.DataFrame(projections)