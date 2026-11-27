"""Bind Experiment 5's key and loss profiles to the shared safeguarded bank."""
from hopfieldpc.gass import GASSBank as CoreGASSBank

from .config import GASSConfig
from .geometry import SafeguardedGeometry
from .reference import MNISTBankProfile


class GASSBank(MNISTBankProfile, CoreGASSBank):
    config_class = GASSConfig
    geometry_class = SafeguardedGeometry
