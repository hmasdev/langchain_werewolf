import json
from typing import Iterable
from pydantic import Field
from .base import BaseGamePlayer
from .helper import runnable_str2game_player_runnable_input
from ..const import GAME_MASTER_NAME
from ..llm_utils import extract_name
from ..enums import ERole, ESide, ESideVictoryCondition
from ..models.state import (
    MsgModel,
    StateModel,
    create_dict_to_record_chat,
)
from ..utils import find_player_by_name


class FortuneTeller(BaseGamePlayer, frozen=True):

    role: ERole = Field(ERole.FortuneTeller, title="the role of the player")
    side: ESide = Field(ESide.Villager, title="the side of the player")
    victory_condition: str = Field(ESideVictoryCondition.VillagersWinCondition.value, title="the victory condition of the player")  # noqa
    night_action: str = Field('Check whether a player is a werewolf or not', title="the night action of the player")  # noqa
    question_to_decide_night_action: str = Field('Who do you want to check whether he/she is a werewolf or not?', title="the question to decide the night action of the player")  # noqa

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
        try:
            target_player = find_player_by_name(target_player_name, players)
            return create_dict_to_record_chat(  # type: ignore # noqa
                self.name,
                [GAME_MASTER_NAME],
                f'{target_player.name} is a werewolf'
                if target_player.role == ERole.Werewolf
                else f'{target_player.name} is not a werewolf'
            )
        except ValueError:
            return create_dict_to_record_chat(  # type: ignore # noqa
                self.name,
                [GAME_MASTER_NAME],
                f'Failed to find the target player: {target_player_name}'
            )
