from typing import Any, Callable, Iterable
from pydantic import BaseModel, Field, ValidationError, field_validator
from ..const import DEFAULT_MODEL, CUSTOM_PLAYER_PREFIX
from ..enums import (
    EInputOutputType,
    ELanguage,
    ESpeakerSelectionMethod,
    ESystemOutputType,
)
from ..game_players.registry import PlayerRoleRegistry
from .state import MsgModel
from ..utils import consecutive_string_generator


class GeneralConfig(BaseModel, frozen=True):
    n_players: int | None = Field(default=None, title="The number of players. Default is None.")  # noqa
    n_players_by_role: dict[str, int] = Field(title="The number of players by role. Default is None.", default_factory=dict)  # noqa
    output: str | None = Field(default=None, title='The output file. Defaults to None.')  # noqa
    system_output_level: ESystemOutputType | str | None = Field(default=None, title=f"The output type of the CLI. {list(ESystemOutputType.__members__.keys())} and player names are valid. Default is None.")  # noqa
    system_output_interface: Callable[[str], None] | EInputOutputType | None = Field(default=None, title="The system output interface. Default is None.")  # noqa
    system_language: ELanguage | None = Field(default=None, title="The system language. Default is None.")  # noqa
    system_formatter: Callable[[MsgModel], str] | str | None = Field(default=None, title="The system formatter. The format should not include anything other than " + ', '.join('"{'+k+'}"' for k in MsgModel.model_fields.keys()))  # noqa
    system_font_color: str | None = Field(default=None, title="The system font color. Default is None.")  # noqa
    player_font_colors: Iterable | str | None = Field(default=None, title="The player font colors. Default is None.")  # noqa
    seed: int | None = Field(default=None, title="The random seed. Defaults to None.")  # noqa
    model: str | None = Field(default=None, title=f"The model to use. Default is None.")  # noqa
    recursion_limit: int | None = Field(default=None, title="The recursion limit. Default is None.")  # noqa
    debug: bool | None = Field(default=None, title="Enable debug mode.")  # noqa
    verbose: bool | None = Field(default=None, title="Enable verbose mode.")  # noqa

    @field_validator("n_players_by_role")
    @classmethod
    def _validate_n_players_by_role(
        cls,
        n_players_by_role: dict[str, int],
    ) -> dict[str, int]:
        # FIXME: comment out because un-commenting this will break the process
        # roles = list(PlayerRoleRegistry.get_keys())
        # for key in n_players_by_role.keys():
        #     if key not in roles:
        #         raise ValidationError(
        #             f"The role {key} is not valid. "
        #             f"The valid roles are: {roles}"
        #             "Please check the role name and try again."
        #             "Or if you want to use a custom role, `PlayerRoleRegistry.register` it first.",  # noqa
        #             [],
        #         )
        return n_players_by_role


class PlayerConfig(BaseModel, frozen=True):
    name: str = Field(title="The name of the player", default_factory=consecutive_string_generator(CUSTOM_PLAYER_PREFIX).__next__)  # noqa
    role: str | None = Field(default=None, title="The role of the player")  # noqa
    model: str = Field(default=DEFAULT_MODEL, title=f"The model to use. Default is {DEFAULT_MODEL}.")  # noqa
    language: ELanguage | None = Field(default=None, title="The language of the player")  # noqa
    player_output_interface: Callable[[str], None] | EInputOutputType | None = Field(default=None, title="The output interface of the player")  # noqa
    player_input_interface: Callable[[str], Any] | EInputOutputType | None = Field(default=None, title="The input interface of the player")  # noqa
    formatter: Callable[[MsgModel], str] | str | None = Field(default=None, title="The formatter of the player. The format should not include anything other than " + ', '.join('"{'+k+'}"' for k in MsgModel.model_fields.keys()))  # noqa

    @field_validator('formatter')
    @classmethod
    def _validate_formatter(
        cls,
        role: str | None = None,
    ) -> str | None:
        if role is None:
            return None
        if role not in PlayerRoleRegistry.get_keys():
            raise ValidationError(
                f"The role {role} is not valid. "
                f"The valid roles are: {list(PlayerRoleRegistry.get_keys())}"
                "Please check the role name and try again."
                "Or if you want to use a custom role, `PlayerRoleRegistry.register` it first.",  # noqa
                [],
            )
        return role


class GameConfig(BaseModel, frozen=True):

    class PreparationConfig(BaseModel, frozen=True):
        game_rule_template: str | None = Field(default=None, title="The game rule template")  # noqa
        role_announce_template: str | None = Field(default=None, title="The role announce template")  # noqa
        role_explanation_template: str | None = Field(default=None, title="The role explanation template")  # noqa

    class CheckVictoryConditionConfig(BaseModel, frozen=True):
        pass

    class ChatConfig(BaseModel, frozen=True):
        prompt: str | None = Field(default=None, title="The prompt of the chat")  # noqa
        system_prompt: str | None = Field(default=None, title="The system prompt of the chat")  # noqa
        select_speaker: ESpeakerSelectionMethod | None = Field(default=None, title="The select speaker method")  # noqa
        n_turns_per_day: int | None = Field(default=None, title="The number of turns per day")  # noqa

    class VoteConfig(BaseModel, frozen=True):
        prompt: str | None = Field(default=None, title="The prompt of the vote")  # noqa
        system_prompt: str | None = Field(default=None, title="The system prompt of the vote")  # noqa
        chat_llm: str | None = Field(default=None, title="The chat LLM used to clean the vote")  # noqa
        seed: int | None = Field(default=None, title="The seed for chat_llm")  # noqa

    class NightActionConfig(BaseModel, frozen=True):
        prompt: str | None = Field(default=None, title="The prompt of the night action")  # noqa

    class EliminationConfig(BaseModel, frozen=True):
        pass

    preparation_kwargs: PreparationConfig = Field(PreparationConfig(), title="The preparation configuration")  # noqa
    check_victory_condition_kwargs: CheckVictoryConditionConfig = Field(CheckVictoryConditionConfig(), title="The check victory condition configuration")  # noqa
    check_victory_condition_before_daytime_kwargs: CheckVictoryConditionConfig = Field(CheckVictoryConditionConfig(), title="The check victory condition configuration (daytime)")  # noqa
    check_victory_condition_before_nighttime_kwargs: CheckVictoryConditionConfig = Field(CheckVictoryConditionConfig(), title="The check victory condition configuration (nighttime)")  # noqa
    chat_kwargs: ChatConfig = Field(ChatConfig(), title="The chat configuration")  # noqa
    daytime_chat_kwargs: ChatConfig = Field(ChatConfig(), title="The chat configuration (daytime)")  # noqa
    nighttime_chat_kwargs: ChatConfig = Field(ChatConfig(), title="The chat configuration (nighttime)")  # noqa
    vote_kwargs: VoteConfig = Field(VoteConfig(), title="The vote configuration")  # noqa
    daytime_vote_kwargs: VoteConfig = Field(VoteConfig(), title="The vote configuration (daytime)")  # noqa
    nighttime_vote_kwargs: VoteConfig = Field(VoteConfig(), title="The vote configuration (nighttime)")  # noqa
    night_action_kwargs: NightActionConfig = Field(NightActionConfig(), title="The night action configuration")  # noqa
    elimination_kwargs: EliminationConfig = Field(EliminationConfig(), title="The elimination configuration")  # noqa
    elimination_after_daytime_vote_kwargs: EliminationConfig = Field(EliminationConfig(), title="The elimination configuration")  # noqa
    elimination_after_night_vote_kwargs: EliminationConfig = Field(EliminationConfig(), title="The elimination configuration")  # noqa


class Config(BaseModel, frozen=True):
    general: GeneralConfig = Field(default=GeneralConfig(), title="The general configuration")  # type: ignore # noqa
    players: list[PlayerConfig] = Field(title="The player configurations", default_factory=list)  # type: ignore # noqa
    game: GameConfig = Field(default=GameConfig(), title="The game configuration")  # type: ignore # noqa
