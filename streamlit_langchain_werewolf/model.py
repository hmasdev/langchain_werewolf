from typing import Any, Callable
from pydantic import BaseModel, Field, field_validator
from langchain_werewolf.const import BASE_LANGUAGE
from langchain_werewolf.enums import EInputOutputType, ELanguage, ESystemOutputType  # noqa
from langchain_werewolf.models.config import Config


class GameSetupModel(BaseModel):

    n_players: int = Field(title='Number of Players', description='Number of players in the game')  # noqa
    n_werewolves: int = Field(title='Number of Werewolves', description='Number of werewolves in the game')  # noqa
    n_knights: int = Field(title='Number of Knights', description='Number of knights in the game')  # noqa
    n_fortune_tellers: int = Field(title='Number of Fortune Tellers', description='Number of fortune tellers in the game')  # noqa

    config: Config = Field(title='Config', description='Game config')  # noqa

    system_input_interface: Callable[[str], Any] | EInputOutputType | None = Field(default=None, title="The interface for input into the system. Default is None.")  # noqa
    system_output_interface: Callable[[Any], None] | EInputOutputType | None = Field(default=None, title="The interface for output from the system. Default is None.")  # noqa

    system_language: ELanguage = Field(default=BASE_LANGUAGE, title='System Language', description='Language to use for the system')  # noqa
    system_output_level: ESystemOutputType | str = Field(default=ESystemOutputType.all, title='System Output Level', description='Output level for the system')  # noqa

    model: str = Field(title='Model', description='Model to use for the game')  # noqa
    seed: int = Field(title='Seed', description='Random seed for the game')  # noqa


class StremlitMessageModel(BaseModel):
    message: str = Field(title='Message', description='Message to send to the client')  # noqa
    token: str = Field(title='Token', description='Token to authenticate the client')  # noqa
    response_required: bool = Field(default=False, title='Response Required', description='Whether a response is required')  # noqa

    @field_validator('message')
    @classmethod
    def clean_message(cls, value: str | None) -> str:
        return value or ''
