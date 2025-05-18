from typing import ClassVar

from ..base import BaseGamePlayerRole
from ..player_sides import WerewolfSideMixin
from ..registry import PlayerRoleRegistry


@PlayerRoleRegistry.register
class Werewolf(BaseGamePlayerRole, WerewolfSideMixin):

    role: ClassVar[str] = 'werewolf'
    night_action: ClassVar[str] = f'Vote exclude a player from the game to help {WerewolfSideMixin.side}'  # noqa
