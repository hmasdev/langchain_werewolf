from typing import ClassVar

from .base import BasePlayerSideMixin
from .registry import PlayerSideRegistry


@PlayerSideRegistry.register
class WerewolfSideMixin(BasePlayerSideMixin):
    """Mixin for Werewolf side"""
    side: ClassVar[str] = "WerewolfSide"
    victory_condition: ClassVar[str] = "The number of alive werewolves equal or outnumber half of the total number of players"  # noqa


@PlayerSideRegistry.register
class VillagerSideMixin(BasePlayerSideMixin):
    """Mixin for Villager side"""
    side: ClassVar[str] = "VillagerSide"
    victory_condition: ClassVar[str] = "All werewolves are excluded from the game"  # noqa


__auto_registered__ = [
    WerewolfSideMixin,
    VillagerSideMixin,
]
