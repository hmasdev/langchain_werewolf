import json
from typing import ClassVar, Iterable
from pydantic import Field
from ..base import BaseGamePlayer, BaseGamePlayerRole
from ..helper import runnable_str2game_player_runnable_input
from ..player_sides import VillagerSideMixin
from ..registry import PlayerRoleRegistry
from ..utils import is_werewolf_role
from ...const import GAME_MASTER_NAME
from ...llm_utils import extract_name
from ...models.state import (
    MsgModel,
    StateModel,
    create_dict_to_record_chat,
)
from ...utils import find_player_by_name


@PlayerRoleRegistry.register
class FortuneTeller(BaseGamePlayerRole, VillagerSideMixin):

    role: ClassVar[str] = 'fortuneteller'
    night_action: ClassVar[str] = 'Check whether a player is a werewolf or not'  # noqa

    question_to_decide_night_action: str = Field(
        default='Who do you want to check whether he/she is a werewolf or not?',  # noqa
        title="the question to decide the night action of the player",
    )

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
                if is_werewolf_role(target_player)
                else f'{target_player.name} is not a werewolf'
            )
        except ValueError:
            return create_dict_to_record_chat(  # type: ignore # noqa
                self.name,
                [GAME_MASTER_NAME],
                f'Failed to find the target player: {target_player_name}'
            )
