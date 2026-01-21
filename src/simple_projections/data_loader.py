"""Load and prepare baseball data using pybaseball"""

import pandas as pd
import numpy as np
from pybaseball import batting_stats, cache
import warnings

warnings.filterwarnings("ignore")
cache.enable()

class DataLoader:

  def __init__(self, min_pa: int = 200):
    self.min_pa = min_pa

  def load_batting_data(self, years):
    """Load batting stats for specified years"""
    all_data = []
    for year in years:
      data = batting_stats(year, qual=self.min_pa)
      data['Season'] = year
      all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

  def clean_and_standardize(self, df):
    """Clean and standardize batting data"""
    df = df.copy()
    df['Name'] = df['Name'].str.strip()

    numeric_cols = ['Age', 'PA', 'AB', 'H', 'HR', 'RBI', 'R', 'SB']
    for col in numeric_cols:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['PA'] >= self.min_pa].copy()
    df = df.dropna(subset=['Name', 'Age', 'PA'])

    return df

  def prepare_for_projections(self, df):
    """Add derived metrics"""
    df = df.copy()

    if 'SLG' in df.columns and 'AVG' in df.columns:
      df['ISO'] = df['SLG'] - df['AVG']

    if 'HR' in df.columns:
      df['HR_rate'] = df['HR'] / df['PA']

    return df

def load_player_data(years, min_pa=200):
  """Convenience function to load and prepare data."""
  loader = DataLoader(min_pa)
  raw_data = loader.load_batting_data(years)
  clean_data = loader.clean_and_standardize(raw_data)

  # Add player IDs
  clean_data['player_id'] = clean_data.groupby('Name').ngroup()

  return loader.prepare_for_projections(clean_data)
