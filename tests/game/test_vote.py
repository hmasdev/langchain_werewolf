import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.enums import ETimeSpan
from langchain_werewolf.game.vote import (
    _player_vote,
)
from langchain_werewolf.game_players.base import BaseGamePlayer
from langchain_werewolf.models.state import MsgModel, StateModel


@pytest.mark.parametrize(
    'timespan',
    [
        ETimeSpan.day,
        ETimeSpan.night,
    ],
)
def test__player_vote(
    timespan: ETimeSpan,
    mocker: MockerFixture,
) -> None:
    # preparation
    expected1 = 'playerX should be excluded'
    expected2 = 'playerX'
    player = mocker.MagicMock(spec=BaseGamePlayer)
    player.name = 'player'
    player.generate_message.return_value = MsgModel(name=player.name, message=expected1)  # noqa
    mocker.patch('langchain_werewolf.game.vote.extract_name', return_value=expected2)  # noqa
    state = StateModel(alive_players_names=[player.name, expected2])
    # execution
    actual = _player_vote(state, timespan, player, generate_system_prompt=str)
    # assert
    print(actual)
    print(state)
    assert actual['chat_state']
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages  # type: ignore # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.message == expected1  # type: ignore # noqa
    if timespan == ETimeSpan.day:
        assert actual['daytime_votes_current'] == {player.name: expected2}
    else:
        assert actual['nighttime_votes_current'] == {player.name: expected2}


def test__player_vote_with_excluded_player(
    mocker: MockerFixture,
) -> None:
    # preparation
    player = mocker.MagicMock(spec=BaseGamePlayer)
    player.name = 'player'
    state = StateModel(alive_players_names=[])
    # execution
    actual = _player_vote(state, ETimeSpan.day, player,  generate_system_prompt=str)  # noqa
    # assert
    assert actual['chat_state'] == {}
