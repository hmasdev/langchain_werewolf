from pydantic import Field
from ..enums import ERole, ESide, ESideVictoryCondition
from .base import BaseGamePlayer


class Villager(BaseGamePlayer, frozen=True):
    role: ERole = Field(ERole.Villager, title="the role of the player")
    side: ESide = Field(ESide.Villager, title="the side of the player")
    victory_condition: str = Field(ESideVictoryCondition.VillagersWinCondition.value, title="the victory condition of the player")  # noqa
