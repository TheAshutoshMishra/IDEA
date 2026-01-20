"""
Source Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Your Team Name"
__description__ = "AadhaarInsight360 - Intelligent Analytics Platform"

from .data_processing import DataPreprocessor
from .pattern_analysis import PatternDetector
from .anomaly_detection import AnomalyDetector
from .predictive_models import Forecaster

__all__ = [
    'DataPreprocessor',
    'PatternDetector',
    'AnomalyDetector',
    'Forecaster'
]
