from collections import namedtuple
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.game_players.base import GamePlayerRunnableInputModel
from langchain_werewolf.game_players.helper import (
    _generate_game_player_runnable_based_on_runnable_lambda,
    _generate_game_player_runnable_based_on_chat_model,
    generate_game_player_runnable,
)


def test__generate_game_player_runnable_based_on_chat_model(
    mocker: MockerFixture,
) -> None:
    # preparation
    input_ = GamePlayerRunnableInputModel(prompt='hello, world', system_prompt=None)  # noqa
    expected = input_.prompt.upper() if isinstance(input_.prompt, str) else input_.prompt.message.upper()  # noqa
    runnable_mock = mocker.MagicMock(spec=BaseChatModel)
    runnable_mock.invoke.return_value = namedtuple('Message', ['content'])(content=expected)  # type: ignore # noqa
    # execution
    actual = _generate_game_player_runnable_based_on_chat_model(runnable_mock)  # noqa
    # assert
    assert actual.InputType == GamePlayerRunnableInputModel
    assert actual.OutputType == str
    assert actual.invoke(input_) == expected


def test__generate_game_player_runnable_based_on_runnable_lambda() -> None:  # noqa
    # preparation
    input_ = GamePlayerRunnableInputModel(prompt='hello, world', system_prompt=None)  # noqa
    expected = input_.prompt.upper() if isinstance(input_.prompt, str) else input_.prompt.message.upper()  # noqa
    runnable_mock = RunnableLambda(str.upper)
    # execution
    actual = _generate_game_player_runnable_based_on_runnable_lambda(runnable_mock)  # noqa
    # assert
    assert actual.InputType == GamePlayerRunnableInputModel
    assert actual.OutputType == str
    assert actual.invoke(input_) == expected


def test_game_player_runnable_with_base_chat_model(mocker: MockerFixture) -> None:  # noqa
    # preparation
    input_ = GamePlayerRunnableInputModel(prompt='hello, world', system_prompt=None)  # noqa
    expected = input_.prompt.upper() if isinstance(input_.prompt, str) else input_.prompt.message.upper()  # noqa
    runnable_mock = mocker.MagicMock(spec=BaseChatModel)
    runnable_mock.invoke.return_value = namedtuple('Message', ['content'])(content=expected)  # type: ignore # noqa
    # execution
    actual = generate_game_player_runnable(runnable_mock)  # noqa
    # assert
    assert actual.InputType == GamePlayerRunnableInputModel | str
    assert actual.OutputType == str
    assert actual.invoke(input_) == expected


def test_game_player_runnable_with_runnable_lambda() -> None:  # noqa
    # preparation
    input_ = GamePlayerRunnableInputModel(prompt='hello, world', system_prompt=None)  # noqa
    expected = input_.prompt.upper() if isinstance(input_.prompt, str) else input_.prompt.message.upper()  # noqa
    runnable_mock = RunnableLambda(str.upper).with_types(input_type=str, output_type=str)  # noqa
    # execution
    actual = generate_game_player_runnable(runnable_mock)  # noqa
    # assert
    assert actual.InputType == GamePlayerRunnableInputModel | str
    assert actual.OutputType == str
    assert actual.invoke(input_) == expected


def test_game_player_runnable_with_invalid_input() -> None:  # noqa
    # assert
    with pytest.raises(ValueError):
        generate_game_player_runnable(None)  # type: ignore
