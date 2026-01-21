"""
PArk factor adjustments for baseball projections.

Park factors account for how different ballparks affect offensive statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")

class ParkFactors:
  """
  Park factor adjustments for projections

  Park factors show how much a park inflates or deflates stats:
  - 1.00 = neutral
  - 1.10 = 10% increase
  - 0.90 = 10% decrease
  """

  def __init__(self):
    self.park_factors = self._get_default_park_factors()

  def _get_default_park_factors(self) -> Dict[str, Dict[str, float]]:
    """
    Get default park factors for 2022-2025 MLB parks, based on multi-year park factor data.

    Returns:
      Dict mapping team/park to stat-specific factors
    """
    factors = {
      # High-offense parks
      'COL': {  # Coors Field - extreme hitter friendly
        'overall': 1.30,
        'HR': 1.25,
        'AVG': 1.15,
        'runs': 1.35
      },
      'CIN': {  # Great American Ball Park
        'overall': 1.12,
        'HR': 1.20,
        'AVG': 1.05,
        'runs': 1.15
      },
      'TEX': {  # Globe Life Field (new)
        'overall': 1.10,
        'HR': 1.15,
        'AVG': 1.05,
        'runs': 1.12
      },
      'BAL': {  # Camden Yards (modified dimensions)
        'overall': 1.08,
        'HR': 1.12,
        'AVG': 1.03,
        'runs': 1.10
      },
      'BOS': {  # Fenway Park
        'overall': 1.06,
        'HR': 1.02,
        'AVG': 1.08,
        'runs': 1.08
      },
      'CHC': {  # Wrigley Field (wind dependent)
        'overall': 1.04,
        'HR': 1.06,
        'AVG': 1.02,
        'runs': 1.05
      },
      'PHI': {  # Citizens Bank Park
        'overall': 1.05,
        'HR': 1.08,
        'AVG': 1.02,
        'runs': 1.06
      },
      'MIL': {  # American Family Field
        'overall': 1.03,
        'HR': 1.05,
        'AVG': 1.01,
        'runs': 1.04
      },
      'NYY': {  # Yankee Stadium (short porch)
        'overall': 1.02,
        'HR': 1.08,
        'AVG': 0.98,
        'runs': 1.03
      },
      'MIN': {  # Target Field
        'overall': 1.02,
        'HR': 1.04,
        'AVG': 1.00,
        'runs': 1.03
      },

      # Neutral parks
      'WSH': {  # Nationals Park
        'overall': 1.00,
        'HR': 1.00,
        'AVG': 1.00,
        'runs': 1.00
      },
      'ATL': {  # Truist Park
        'overall': 1.00,
        'HR': 1.02,
        'AVG': 0.99,
        'runs': 1.00
      },
      'STL': {  # Busch Stadium
        'overall': 0.99,
        'HR': 0.98,
        'AVG': 1.00,
        'runs': 0.99
      },
      'KC': {  # Kauffman Stadium
        'overall': 0.98,
        'HR': 0.95,
        'AVG': 1.00,
        'runs': 0.97
      },
      'ARI': {  # Chase Field
        'overall': 1.01,
        'HR': 1.03,
        'AVG': 1.00,
        'runs': 1.02
      },
      'CLE': {  # Progressive Field
        'overall': 0.98,
        'HR': 0.96,
        'AVG': 0.99,
        'runs': 0.97
      },
      'DET': {  # Comerica Park
        'overall': 0.96,
        'HR': 0.92,
        'AVG': 0.98,
        'runs': 0.95
      },
      'HOU': {  # Minute Maid Park
        'overall': 0.99,
        'HR': 1.00,
        'AVG': 0.98,
        'runs': 0.99
      },
      'LAD': {  # Dodger Stadium
        'overall': 0.97,
        'HR': 0.95,
        'AVG': 0.98,
        'runs': 0.96
      },
      'NYM': {  # Citi Field
        'overall': 0.96,
        'HR': 0.92,
        'AVG': 0.98,
        'runs': 0.95
      },
      'TOR': {  # Rogers Centre
        'overall': 1.00,
        'HR': 1.02,
        'AVG': 0.99,
        'runs': 1.00
      },
      'TB': {  # Tropicana Field
        'overall': 0.97,
        'HR': 0.96,
        'AVG': 0.98,
        'runs': 0.96
      },
      'PIT': {  # PNC Park
        'overall': 0.96,
        'HR': 0.94,
        'AVG': 0.97,
        'runs': 0.95
      },
      'LAA': {  # Angel Stadium
        'overall': 0.98,
        'HR': 0.98,
        'AVG': 0.98,
        'runs': 0.97
      },
      'CHW': {  # Guaranteed Rate Field
        'overall': 1.01,
        'HR': 1.04,
        'AVG': 0.99,
        'runs': 1.02
      },

      # Pitcher-friendly parks
      'OAK': {  # Oakland Coliseum
        'overall': 0.93,
        'HR': 0.88,
        'AVG': 0.96,
        'runs': 0.92
      },
      'MIA': {  # LoanDepot Park
        'overall': 0.92,
        'HR': 0.88,
        'AVG': 0.94,
        'runs': 0.90
      },
      'SEA': {  # T-Mobile Park
        'overall': 0.93,
        'HR': 0.88,
        'AVG': 0.96,
        'runs': 0.92
      },
      'SF': {  # Oracle Park - most pitcher friendly
        'overall': 0.90,
        'HR': 0.82,
        'AVG': 0.94,
        'runs': 0.88
      },
      'SD': {  # Petco Park
        'overall': 0.92,
        'HR': 0.86,
        'AVG': 0.96,
        'runs': 0.91
      },
    }

    return factors

  def get_park_factor(self, team: str, stat: str = 'overall') -> float:
    """
    Get park factor for team/stat combination.

    Args:
      team: Team abbreviation
      stat: Specific stat ('HR', 'AVG', 'runs', 'overall')

    Returns:
      Park factor (1.0 = neutral)
    """
    if team not in self.park_factors:
      return 1.0

    park_data = self.park_factors[team]

    if stat in park_data:
      return park_data[stat]
    else:
      return park_data.get('overall', 1.0)

  def adjust_stat(self, value: float, from_team: str, to_team: str, stat: str = 'overall') -> float:
    """
    Adjust a stat for moving between parks.

    Args:
      value: original stat value
      from_team: Current team
      to_team: New team
      stat: Type of stat

    Returns:
      Park-adjusted value
    """
    from_factor = self.get_park_factor(from_team, stat)
    to_factor = self.get_park_factor(to_team, stat)

    # Neutralize from old park, apply new park
    neutral_value = value / from_factor
    adjusted_value = neutral_value * to_factor

    return adjusted_value

  def neutralize_stats(self, stats: pd.Series, team: str) -> pd.Series:
    """
    Convert park-specific stats to park-neutral.

    Args:
      stats: Player stats
      team: Team player played for

    Returns:
      Park-neutralized stats
    """
    neutralized = stats.copy()

    # Adjust counting stats
    if 'HR' in neutralized.index:
      hr_factor = self.get_park_factor(team, 'HR')
      neutralized['HR'] = neutralized['HR'] / hr_factor

    # Adjust rate stats
    if 'AVG' in neutralized.index:
      avg_factor = self.get_park_factor(team, 'AVG')
      neutralized['AVG'] = neutralized['AVG'] / avg_factor

    if 'OBP' in neutralized.index:
      obp_factor = self.get_park_factor(team, 'overall')
      neutralized['OBP'] = neutralized['OBP'] / obp_factor

    if 'SLG' in neutralized.index:
      slg_factor = (self.get_park_factor(team, 'HR') +
                    self.get_ark_factor(team, 'AVG')) / 2
      neutralized['SLG'] = neutralized['SLG'] / slg_factor

    if 'wOBA' in neutralized.index:
      woba_factor = self.get_park_factor(team, 'overall')
      neutralized['wOBA'] = neutralized['wOBA'] / woba_factor

    return neutralized

  def apply_projections_park_adjustments(self, projections: pd.DataFrame, team_assignments: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Apply park factors to projections

    Args:
      projections: DataFrame with projections
      team_assignments: Dict mapping player names to teams

    Returns:
      Park-adjusted projections
    """
    adjusted = projections.copy()

    for idx, row in adjusted.iterrows():
      if team_assignments and row['Name'] in team_assignments:
        team = team_assignments[row['Name']]
      elif 'Team' in row.index:
        team = row['Team']
      else:
        continue

      if 'HR' in row.index:
        hr_factor = self.get_park_factor(team, 'HR')
        adjusted.at[idx, 'HR'] = row['HR'] * hr_factor

      if 'AVG' in row.index:
        avg_factor = self.get_park_factor(team, 'AVG')
        adjusted.at[idx, 'AVG'] = row['AVG'] * avg_factor

      if 'wOBA' in row.index:
        woba_factor = self.get_park_factor(team, 'overall')
        adjusted.at[idx, 'wOBA'] = row['wOBA'] * woba_factor

    return adjusted

  def get_park_rankings(self, stat: str = 'overall') -> pd.DataFrame:
    """
    Get rankings of perks by seat

    Args:
      stat: Stat to rank by

    Returns:
      DataFrame with park rankings
    """
    rankings = []

    for team, factors in self.park_factors.items():
      factor = factors.get(stat, factors.get('overall', 1.0))
      rankings.append({
        'Team': team,
        'Park Factor': factor,
        'Effect': 'Hitter Friendly' if factor > 1.0 else
                  'Pitcher Friendly' if factor < 1.0 else 'Neutral'
      })

    df = pd.DataFrame(rankings)
    df = df.sort_values('Park Factor', ascending=False)

    return df