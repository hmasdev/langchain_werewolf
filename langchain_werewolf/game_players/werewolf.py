from pydantic import Field
from ..enums import ERole, ESide, ESideVictoryCondition
from .base import BaseGamePlayer


class Werewolf(BaseGamePlayer, frozen=True):

    role: ERole = Field(ERole.Werewolf, title="the role of the player")
    side: ESide = Field(ESide.Werewolf, title="the side of the player")
    victory_condition: str = Field(ESideVictoryCondition.WerewolvesWinCondition.value, title="the victory condition of the player")  # noqa
    night_action: str = Field(f'Vot exclude a player from the game to help {ESide.Werewolf}', title="the night action of the player")  # noqa
