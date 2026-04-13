"""
Shared utilities for the Mechanistic Digital Twin.
"""

"""
    years_to_hours(t_years)

Convert time from years to hours for bridging clinical and molecular timescales.
"""
years_to_hours(t_years) = t_years * 365.25 * 24.0

"""
    hours_to_years(t_hours)

Convert time from hours to years.
"""
hours_to_years(t_hours) = t_hours / (365.25 * 24.0)
