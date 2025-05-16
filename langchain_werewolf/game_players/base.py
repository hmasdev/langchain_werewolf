from typing import Callable, ClassVar, Iterable

from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from pydantic import (
    BaseModel,
    Field,
    SkipValidation,
    field_validator,
    ConfigDict,
)

from ..models.state import (
    MsgModel,
    StateModel,
    create_dict_without_state_updated,
    get_related_messsages,
)


_DEFAULT_FORMATTER = MsgModel.format


class GamePlayerRunnableInputModel(BaseModel):
    prompt: str | MsgModel = Field(..., title="the prompt to generate the message")  # noqa
    system_prompt: str | None = Field(default=None, title="the system prompt to generate the message")  # noqa


class BaseGamePlayer(BaseModel, frozen=True):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., title="the name of the player")
    runnable: SkipValidation[Runnable[GamePlayerRunnableInputModel, str]] = Field(
        title="a runnable to define player's behavior",
        description="a runnable to define player's behavior which receives prompt and sysmtem prompt from the game and returns a player's message",  # noqa
    )

    # NOTE: output and formatter will be used to output messages to the player, like through a console.  # noqa
    output: SkipValidation[Runnable[str, None] | None] = Field(
        default=None,
        title="the output function to display a message to the player",
        description="the output function to display a message to the player through a console",  # noqa
    )
    formatter: Callable[[MsgModel], str] | str | None = Field(
        default=None,
        title="the formatter which will be used to format the message before output",  # noqa
        description="the formatter which will be used to format the message before output. "  # noqa
    )

    translator: SkipValidation[Runnable[str, str]] = Field(
        default=RunnablePassthrough(),
        title="the translator used to translate the game language to player's language",  # noqa
    )
    inv_translator: SkipValidation[Runnable[str, str]] = Field(
        default=RunnablePassthrough(),
        title="the translator used to translate player's message to the game language",  # noqa
    )

    @field_validator('output')
    @classmethod
    def _preprocess_output(
        cls,
        output: Callable[[str], None] | Runnable[str, None] | None = None,
    ) -> Runnable[str, None] | None:
        if output is not None and not isinstance(output, Runnable):
            return RunnableLambda(output)
        return output

    @field_validator('formatter')
    @classmethod
    def _validate_formatter(
        cls,
        formatter: Callable[[MsgModel], str] | str | None = None,
    ) -> Callable[[MsgModel], str] | str:
        formatter = formatter or _DEFAULT_FORMATTER
        if isinstance(formatter, str):
            try:
                # test
                formatter.format(**MsgModel(name='name', message='message').model_dump())  # noqa
            except KeyError:
                raise ValueError(
                    "The formatter should not includes anything other than "
                    + ', '.join('"{'+k+'}"' for k in MsgModel.model_fields.keys())  # noqa
                    + '. But the formatter is '
                    + f'"{formatter}".'
                )
        return formatter

    def receive_message(self, message: MsgModel) -> None:
        """Show the player

        Args:
            message (MsgModel): the message to show
        """
        formatter = self.formatter or _DEFAULT_FORMATTER
        if self.output:
            if isinstance(formatter, str):
                self.output.invoke(formatter.format(**message.model_dump()))  # noqa
            else:
                self.output.invoke(formatter(message))

    def generate_message(
        self,
        prompt: str | MsgModel,
        system_prompt: str | None = None,
    ) -> MsgModel:
        """Generate a message

        Args:
            prompt (str | MsgModel): the prompt to generate the message
            system_prompt (str | None, optional): the system prompt to generate the message. Defaults to None.

        Returns:
            MsgModel: the generated message
        """  # noqa
        return MsgModel(
            name=self.name,
            message=self.runnable.invoke(GamePlayerRunnableInputModel(
                prompt=prompt,
                system_prompt=system_prompt,
            ))
        )

    def act_in_night(
        self,
        players: Iterable["BaseGamePlayer"],
        messages: Iterable[MsgModel],
        state: StateModel,
    ) -> dict[str, object]:
        f"""Player's action in the night

        Args:
            players (Iterable[TBaseGamePlayer]): all players
            messages (Iterable[MsgModel]): all messages
            state (StateModel): the global state

        Returns:
            dict[str, object]: dict to update the state

        Note:
            the argument 'messages' should be generated by `langchain_werewolf.models.state.get_related_messsages`
            the argument 'state' should be filtered according to the player role and side`
        """  # noqa
        # FIXME: players implement act_in_night to know anything about the game
        #        because the argument may include all players information, all messages, and the global state  # noqa
        return create_dict_without_state_updated(state)


class BaseGamePlayerRole(BaseGamePlayer):

    def __init_subclass__(cls):
        """Enforce the implementation of night_action and role"""
        super().__init_subclass__()

        if not hasattr(cls, "role"):
            raise NotImplementedError(f"The class `{cls.__name__}` does not implement `role`")  # noqa
        if not hasattr(cls, "night_action"):
            raise NotImplementedError(f"The class `{cls.__name__}` does not implement `night_action`")  # noqa
        if not isinstance(cls.role, str):
            raise TypeError(f"The class `{cls.__name__}` should implement the `role` attribute as str")  # noqa
        if not isinstance(cls.night_action, str):
            raise TypeError(f"The class `{cls.__name__}` should implement the `night_action` attribute as str")  # noqa

    role: ClassVar[str]
    night_action: ClassVar[str]


class BasePlayerSideMixin:
    """
    Base class for player side which enforces the implementation of side and victory_condition
    """  # noqa

    def __init_subclass__(cls):
        """Enforce the implementation of side and victory_condition

        Raises:
            NotImplementedError: if the class does not implement side or victory_condition
            TypeError: if the class does not implement side or victory_condition as str
        """  # noqa
        super().__init_subclass__()
        if not hasattr(cls, 'side'):
            raise NotImplementedError(f"The class `{cls.__name__}` should implement the `side` attribute")  # noqa
        if not hasattr(cls, 'victory_condition'):
            raise NotImplementedError(f"The class `{cls.__name__}` should implement the `victory_condition` attribute")  # noqa
        if not isinstance(cls.side, str):
            raise TypeError(f"The class `{cls.__name__}` should implement the `side` attribute as str")  # noqa
        if not isinstance(cls.victory_condition, str):
            raise TypeError(f"The class `{cls.__name__}` should implement the `victory_condition` attribute as str")  # noqa

    side: str
    victory_condition: str
