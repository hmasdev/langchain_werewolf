from typing import Final

from .base import BasePlayerSideMixin
from .registry import PlayerSideRegistry


@PlayerSideRegistry.register
class WerewolfSideMixin(BasePlayerSideMixin):
    """Mixin for Werewolf side"""
    side: Final[str] = "WerewolfSide"
    victory_condition: Final[str] = "The number of alive werewolves equal or outnumber half of the total number of players"


@PlayerSideRegistry.register
class VillagerSideMixin(BasePlayerSideMixin):
    """Mixin for Villager side"""
    side: Final[str] = "VillagerSide"
    victory_condition: Final[str] = "All werewolves are excluded from the game"


__auto_registered__ = [
    WerewolfSideMixin,
    VillagerSideMixin,
]
