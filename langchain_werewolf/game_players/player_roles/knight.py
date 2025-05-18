import json
from typing import ClassVar, Iterable
from langchain_core.exceptions import OutputParserException
from pydantic import Field
from ..base import BaseGamePlayer, BaseGamePlayerRole
from ...const import GAME_MASTER_NAME
from ..player_sides import VillagerSideMixin
from ..registry import PlayerRoleRegistry
from ...llm_utils import extract_name
from ...models.state import (
    MsgModel,
    StateModel,
    create_dict_to_add_safe_player,
    create_dict_to_record_chat,
)


@PlayerRoleRegistry.register
class Knight(BaseGamePlayerRole, VillagerSideMixin):

    role: ClassVar[str] = 'knight'
    night_action: ClassVar[str] = 'Save a player from the werewolves'

    question_to_decide_night_action: str = Field(
        default='Who do you want to save in this night?',
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
        try:
            target_player_name = extract_name(
                target_player_name_raw.message,
                [p.name for p in players if p.name in state.alive_players_names and p.name != self.name],  # noqa
                context=f'Extract the valid name of the player as the answer to "{self.question_to_decide_night_action}"',  # noqa
                chat_model=self.runnable,
            )
        except OutputParserException:
            return create_dict_to_record_chat(  # type: ignore # noqa
                self.name,
                [GAME_MASTER_NAME],
                'Failed to decide the target player.',
            )
        return (  # type: ignore
            create_dict_to_add_safe_player(target_player_name)
            | create_dict_to_record_chat(  # type: ignore # noqa
                self.name,
                [GAME_MASTER_NAME],
                f'I decided to save {target_player_name} in this night.',
            )
        )
