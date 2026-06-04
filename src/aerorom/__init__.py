"""
Aerodynamic Reduced Order Model Identification Tool (AeroROM)

This package provides tools for identifying state-space models of aerodynamic
forces from CFD or experimental data for flutter analysis.

Modules:
    preprocessing - Data loading, cleaning, and preprocessing
    identification - System identification algorithms (ERA, ARX, etc.)
    validation - Model validation and error analysis
    export - Model export for external use
    visualization - Data and results visualization
"""

__version__ = '1.0.0'

from .preprocessing import DataPreprocessor
from .identification import ERA, ARX, StateSpaceModel, estimate_max_frequency, check_frequency_characteristics
from .validation import ModelValidator
from .export import ModelExporter
from .visualization import Plotter
from .flutter_analysis import (
    FlutterAnalysis,
    create_standard_section_model,
    create_flutter_demo_rom
)

__all__ = [
    'DataPreprocessor',
    'ERA',
    'ARX',
    'StateSpaceModel',
    'ModelValidator',
    'ModelExporter',
    'Plotter',
    'FlutterAnalysis',
    'create_standard_section_model',
    'create_flutter_demo_rom',
    'estimate_max_frequency',
    'check_frequency_characteristics',
]
