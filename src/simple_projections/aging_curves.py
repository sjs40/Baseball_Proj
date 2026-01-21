"""Aging curves for player projections"""

import pandas as pd
import numpy as np

class AgingCurve:
  """Age adjustment factors based on research, codified into constants"""

  PEAK_POWER = 27
  PEAK_CONTACT = 28
  PEAK_SPEED = 24

  def __init__(self):
    self.power_curve = self._create_power_curve()
    self.contact_curve = self._create_contact_curve()
    self.speed_curve = self._create_speed_curve()

  def _create_power_curve(self):
    """Power aging curve (HR, ISO, SLG)"""
    curve = {}
    for age in range(20, 45):
      if age < 22:
        curve[age] = 0.90 + (age - 20) * 0.03
      elif age <= 27:
        curve[age] = 0.96 + (age - 22) * 0.016
      elif age <= 30:
        curve[age] = 1.04 - (age - 27) * 0.015
      elif age <= 35:
        curve[age] = 0.995 - (age - 30) * 0.025
      else:
        curve[age] = 0.870 - (age - 35) * 0.03
    return curve

  def _create_contact_curve(self):
    """Contact/AVG aging curve"""
    curve = {}
    for age in range(20, 45):
      if age < 23:
        curve[age] = 0.96 + (age - 20) * 0.01
      elif age <= 28:
        curve[age] = 0.99 + (age - 23) * 0.004
      elif age <= 32:
        curve[age] = 1.01 - (age - 28) * 0.005
      elif age <= 36:
        curve[age] = 0.99 - (age - 32) * 0.01
      else:
        curve[age] = 0.95 - (age - 36) * 0.015
    return curve

  def _create_speed_curve(self):
    """Speed aging curve (SB)"""
    curve = {}
    for age in range(20, 45):
      if age <= 24:
        curve[age] = 1.00 + (24 - age) * 0.01
      elif age <= 28:
        curve[age] = 1.00 - (age - 24) * 0.02
      elif age <= 33:
        curve[age] = 0.92 - (age - 28) * 0.025
      else:
        curve[age] = 0.795 - (age - 33) * 0.035
    return curve

  def get_power_adjustment(self, age):
    return self.power_curve.get(age, 0.85)

  def get_contact_adjustment(self, age):
    return self.contact_curve.get(age, 0.90)

  def get_speed_adjustment(self, age):
    return self.speed_curve.get(age, 0.70)

  def apply_aging_to_stats(self, stats, current_age, projected_age):
    """Apply aging adjustments to stats"""
    if projected_age == current_age:
      return stats

    adjusted = stats.copy()

    # Power adjustment
    power_adj = (self.get_power_adjustment(projected_age) / self.get_power_adjustment(current_age))

    for stat in ['HR', 'HR_rate', 'ISO', 'SLG']:
      if stat in adjusted.index:
        adjusted[stat] *= power_adj

    # Contact adjustment
    contact_adj = (self.get_contact_adjustment(projected_age) / self.get_contact_adjustment(current_age))

    if 'AVG' in adjusted.index:
      adjusted['AVG'] *= contact_adj

    # Speed adjustment
    speed_adj = (self.get_speed_adjustment(projected_age) / self.get_speed_adjustment(current_age))

    for stat in ['SB', 'SB_rate']:
      if stat in adjusted.index:
        adjusted[stat] *= speed_adj

    return adjusted