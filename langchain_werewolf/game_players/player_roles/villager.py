from typing import Final

from ..base import BaseGamePlayerRole
from ..player_sides import VillagerSideMixin
from ..registry import PlayerRoleRegistry


@PlayerRoleRegistry.register
class Villager(BaseGamePlayerRole, VillagerSideMixin, frozen=True):

    role: Final[str] = 'villager'
    night_action: Final[str] = 'No night action'
