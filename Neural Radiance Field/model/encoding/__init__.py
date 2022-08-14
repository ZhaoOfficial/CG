from encoding import Encoder
from trigonometric import TrihonometricEncoder
from utils.utils import string_case_insensitive

def make_encoder(config: dict) -> Encoder:
    name = config["name"]
    if string_case_insensitive(name, "Trihonometric"):
        return TrihonometricEncoder(config)
    else:
        raise KeyError(
            "Wrong encoder name: got `{}`.".format(name)
        )
