import json
from typing import Iterable
from pydantic import Field
from .base import BaseGamePlayer
from .helper import runnable_str2game_player_runnable_input
from ..llm_utils import extract_name
from ..enums import ERole, ESide, ESideVictoryCondition
from ..models.state import (
    MsgModel,
    StateModel,
    create_dict_to_add_safe_player,
)


class Knight(BaseGamePlayer, frozen=True):

    role: ERole = Field(default=ERole.Knight, title="the role of the player")
    side: ESide = Field(default=ESide.Villager, title="the side of the player")  # noqa
    victory_condition: str = Field(default=ESideVictoryCondition.VillagersWinCondition.value, title="the victory condition of the player")  # noqa
    night_action: str = Field(default='Save a player from the werewolves', title="the night action of the player")  # noqa
    question_to_decide_night_action: str = Field(default='Who do you want to save in this night?', title="the question to decide the night action of the player")  # noqa

    def act_in_night(
        self,
        players: Iterable[BaseGamePlayer],
        messages: Iterable[MsgModel],
        state: StateModel,
    ) -> dict[str, object]:
        target_player_name_raw = self.generate_message(
            prompt=self.question_to_decide_night_action,
            system_prompt=json.dumps([m.model_dump() for m in messages]),
        )
        target_player_name = extract_name(
            target_player_name_raw.message,
            [p.name for p in players if p.name in state.alive_players_names],  # noqa
            context=f'Extract the valid name of the player as the answer to "{self.question_to_decide_night_action}"',  # noqa
            chat_model=runnable_str2game_player_runnable_input | self.runnable,  # noqa
        )
        return create_dict_to_add_safe_player(target_player_name)  # type: ignore # noqa
