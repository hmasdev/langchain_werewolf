from operator import attrgetter
from typing import Callable
from langchain_core.runnables import Runnable
from pydantic import ValidationError
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.game_players.base import (
    _DEFAULT_FORMATTER,
    BaseGamePlayer,
    BaseGamePlayerRole,
    BasePlayerSideMixin,
    GamePlayerRunnableInputModel,
)
from langchain_werewolf.models.state import MsgModel

# TODO: Add tests for BaseGamePlayer's methods


_DUMMY_MESSAGE = MsgModel(name='name', message='message')


def test_BaseGamePlayer_receive_message_with_default_formatter(
    mocker: MockerFixture,
) -> None:
    runnable_mock = mocker.MagicMock(spec=Runnable[GamePlayerRunnableInputModel, str])  # noqa
    output_mock = mocker.MagicMock(spec=Runnable[str, None])
    player = BaseGamePlayer(
        name='name',
        runnable=runnable_mock,
        output=output_mock,
    )
    player.receive_message(_DUMMY_MESSAGE)
    output_mock.assert_called_once_with(_DEFAULT_FORMATTER(_DUMMY_MESSAGE))


@pytest.mark.parametrize(
    'formatter',
    [
        MsgModel.format,
        attrgetter('message'),
    ]
)
def test_BaseGamePlayer_receive_message_with_callable_foratter(
    formatter: Callable[[MsgModel], str],
    mocker: MockerFixture,
) -> None:
    runnable_mock = mocker.MagicMock(spec=Runnable[GamePlayerRunnableInputModel, str])  # noqa
    output_mock = mocker.MagicMock(spec=Runnable[str, None])
    player = BaseGamePlayer(
        name='name',
        runnable=runnable_mock,
        output=output_mock,
        formatter=formatter,
    )
    player.receive_message(_DUMMY_MESSAGE)
    output_mock.assert_called_once_with(formatter(_DUMMY_MESSAGE))


@pytest.mark.parametrize(
    'formatter',
    [
        '{name}',
        '{message}',
        '{participants}',
        '{timestamp}',
        '{name}: {message}',
        '{timestamp}: {name}: {message}',
        '{timestamp}: {name}: {message}: {participants}',
    ]
)
def test_BaseGamePlayer_receive_message_with_str_formatter(
    formatter: str,
    mocker: MockerFixture,
) -> None:
    runnable_mock = mocker.MagicMock(spec=Runnable[GamePlayerRunnableInputModel, str])  # noqa
    output_mock = mocker.MagicMock(spec=Runnable[str, None])
    player = BaseGamePlayer(
        name='name',
        runnable=runnable_mock,
        output=output_mock,
        formatter=formatter,
    )
    player.receive_message(_DUMMY_MESSAGE)
    output_mock.assert_called_once_with(formatter.format(**_DUMMY_MESSAGE.model_dump()))  # noqa


@pytest.mark.parametrize(
    'formatter',
    [
        '{not_exist}:{name}',
    ]
)
def test_BaseGamePlayer_receive_message_with_invalid_formatter(
    formatter: str,
    mocker: MockerFixture,
) -> None:
    runnable_mock = mocker.MagicMock(spec=Runnable[GamePlayerRunnableInputModel, str])  # noqa
    output_mock = mocker.MagicMock(spec=Runnable[str, None])
    with pytest.raises(ValidationError):
        BaseGamePlayer(
            name='name',
            runnable=runnable_mock,
            output=output_mock,
            formatter=formatter
        )


def test_BaseGamePlayerRole_enforce_attributes_implementation() -> None:

    # Positive test
    class WithRoleWithNightAction(BaseGamePlayerRole):
        role: str = 'role'
        night_action: str = 'night_action'

    # Negative test
    with pytest.raises(NotImplementedError):
        class WithRoleWithoutNightAction(BaseGamePlayerRole):
            role: str = 'role'

    with pytest.raises(NotImplementedError):
        class WithoutRoleWithNightAction(BaseGamePlayerRole):
            night_action: str = 'night_action'

    with pytest.raises(TypeError):
        class WithRoleWithInvalidNightAction(BaseGamePlayerRole):
            role: str = 'role'
            night_action: int = 1

    with pytest.raises(TypeError):
        class WithInvalidRoleWithNightAction(BaseGamePlayerRole):
            role: int = 1
            night_action: str = 'night_action'


def test_BasePlayerSideMixin_enforce_attributes_implementation() -> None:

    # Positive test
    class WithPlayerSide(BasePlayerSideMixin):
        side: str = 'side'
        victory_condition: str = 'victory_condition'

    # Negative test
    with pytest.raises(NotImplementedError):
        class WithPlayerSideWithoutVictoryCondition(BasePlayerSideMixin):
            side: str = 'side'

    with pytest.raises(NotImplementedError):
        class WithoutPlayerSideWithVictoryCondition(BasePlayerSideMixin):
            victory_condition: str = 'victory_condition'

    with pytest.raises(TypeError):
        class WithPlayerSideWithInvalidVictoryCondition(BasePlayerSideMixin):
            side: str = 'side'
            victory_condition: int = 1

    with pytest.raises(TypeError):
        class WithInvalidPlayerSideWithVictoryCondition(BasePlayerSideMixin):
            side: int = 1
            victory_condition: str = 'victory_condition'
