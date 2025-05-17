from typing import ClassVar

from ..base import BaseGamePlayerRole
from ..player_sides import VillagerSideMixin
from ..registry import PlayerRoleRegistry


@PlayerRoleRegistry.register
class Villager(BaseGamePlayerRole, VillagerSideMixin):

    role: ClassVar[str] = 'villager'
    night_action: ClassVar[str] = 'No night action'
