from .fortune_teller import FortuneTeller
from .knight import Knight
from .villager import Villager
from .werewolf import Werewolf

__all__ = [
    FortuneTeller.__name__,
    Knight.__name__,
    Villager.__name__,
    Werewolf.__name__,
]

__auto_registered__ = [
    FortuneTeller,
    Knight,
    Villager,
    Werewolf,
]
