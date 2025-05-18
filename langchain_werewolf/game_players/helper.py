from operator import attrgetter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from .base import GamePlayerRunnableInputModel
from .utils import is_werewolf_role
from ..models.state import MsgModel

runnable_str2game_player_runnable_input: Runnable[
    str,
    GamePlayerRunnableInputModel,
] = RunnableLambda(
    lambda s: GamePlayerRunnableInputModel(prompt=s, system_prompt=None)
)


def _generate_game_player_runnable_based_on_chat_model(
    chat_model: BaseChatModel,
) -> Runnable[GamePlayerRunnableInputModel, str]:
    return (
        RunnableLambda(lambda input: [
            SystemMessage(content=input.system_prompt or ''),
            HumanMessage(content=input.prompt if isinstance(input.prompt, str) else input.prompt.message),  # noqa
        ])
        | chat_model
        | RunnableLambda(attrgetter('content'))
    ).with_types(
        input_type=GamePlayerRunnableInputModel,
        output_type=str,
    )


def _generate_game_player_runnable_based_on_runnable_lambda(
    runnable: Runnable[str, str],
) -> Runnable[GamePlayerRunnableInputModel, str]:
    return (
        RunnableLambda(attrgetter('prompt'))
        | RunnableBranch(
            (
                RunnableLambda(lambda x: isinstance(x, MsgModel)),
                RunnableLambda(lambda x: x.message),
            ),
            RunnablePassthrough(),
        ).with_types(
            input_type=MsgModel,
            output_type=str,  # type: ignore
        )
        | runnable
    ).with_types(
        input_type=GamePlayerRunnableInputModel,
        output_type=str,
    )


def generate_game_player_runnable(
    chatmodel_or_runnable: BaseChatModel | Runnable[str, str],
) -> Runnable[GamePlayerRunnableInputModel, str]:
    """Generate a runnable for BaseGamePlayer.runnable

    Args:
        chatmodel_or_runnable (BaseChatModel | Runnable[str, str]): base chat model or runnable

    Raises:
        ValueError: raise if chatmodel_or_runnable is not a BaseChatModel or Runnable[str, str]

    Returns:
        Runnable[GamePlayerRunnableInputModel, str]: the runnable for BaseGamePlayer.runnable
    """  # noqa

    if isinstance(chatmodel_or_runnable, BaseChatModel):
        return _generate_game_player_runnable_based_on_chat_model(chatmodel_or_runnable)  # noqa
    elif all([
        isinstance(chatmodel_or_runnable, Runnable),
        hasattr(chatmodel_or_runnable, 'InputType') and chatmodel_or_runnable.InputType == str,  # noqa
        hasattr(chatmodel_or_runnable, 'OutputType') and chatmodel_or_runnable.OutputType == str,  # noqa
    ]):
        return _generate_game_player_runnable_based_on_runnable_lambda(chatmodel_or_runnable)  # noqa
    else:
        raise ValueError(f'chatmodel_or_runnable must be either a BaseChatModel or a Runnable[str, str] but {chatmodel_or_runnable}')  # noqa
