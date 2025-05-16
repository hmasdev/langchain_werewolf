from typing import Final

from ..base import BaseGamePlayerRole
from ..player_sides import WerewolfSideMixin
from ..registry import PlayerRoleRegistry


@PlayerRoleRegistry.register
class Werewolf(BaseGamePlayerRole, WerewolfSideMixin, frozen=True):

    role: Final[str] = 'werewolf'
    night_action: Final[str] = f'Vote exclude a player from the game to help {WerewolfSideMixin.side}'  # noqa
