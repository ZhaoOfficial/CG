from .encoding import Encoder
from .trigonometric import TrigonometricEncoder
from .spherical_harmonic import SphericalHarmonicEncoder
from utils.utils import string_case_insensitive

def make_encoder(config: dict) -> Encoder:
    name = config["name"]
    if string_case_insensitive(name, "Trigonometric"):
        return TrigonometricEncoder(config)
    elif string_case_insensitive(name, "SphericalHarmonics"):
        return SphericalHarmonicEncoder(config)
    else:
        raise KeyError("Wrong encoder name: got `{}`.".format(name))
