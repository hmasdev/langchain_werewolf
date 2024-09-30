import pytest
from langchain_werewolf.enums import ETimeSpan
from langchain_werewolf.game.elimination import (
    _eliminate_player,
)
from langchain_werewolf.models.general import IdentifiedModel
from langchain_werewolf.models.state import StateModel


@pytest.mark.parametrize(
    'votes, timespan, expected',
    [
        (
            {'player1': 'player2', 'player2': 'player1', 'player3': 'player1'},
            ETimeSpan.day,
            'player1',
        ),
        (
            {'player1': 'player2', 'player2': 'player1', 'player3': 'player1'},
            ETimeSpan.night,
            'player1',
        ),
        (
            {'player1': 'player2', 'player2': 'player3', 'player3': 'player2'},
            ETimeSpan.day,
            'player2',
        ),
        (
            {'player1': 'player2', 'player2': 'player3', 'player3': 'player2'},
            ETimeSpan.night,
            'player2',
        ),
    ],
)
def test__eliminate_player(
    votes: dict[str, str],
    timespan: ETimeSpan,
    expected: str,
) -> None:
    # preparation
    state = StateModel(alive_players_names=list(votes.keys()), timespan=timespan)  # noqa
    if timespan == ETimeSpan.day:
        state.daytime_votes_history = [IdentifiedModel(value=votes)]
    else:
        state.nighttime_votes_history = [IdentifiedModel(value=votes)]
    # execution
    actual = _eliminate_player(state)
    # assert
    assert actual['alive_players_names'] == [n for n in state.alive_players_names if n != expected]  # noqa
    if timespan == ETimeSpan.day:
        assert actual['daytime_vote_result_history'][-1].value == expected
    else:
        assert actual['nighttime_vote_result_history'][-1].value == expected


@pytest.mark.parametrize(
    'timespan',
    [
        ETimeSpan.day,
        ETimeSpan.night,
    ],
)
def test__eliminate_player_without_valid_votes(
    timespan: ETimeSpan,
) -> None:
    # preparation
    votes = {'player1': 'hoge', 'player2': 'fuga', 'player3': 'piyo'}
    state = StateModel(alive_players_names=list(votes.keys()), timespan=timespan)  # noqa
    if timespan == ETimeSpan.day:
        state.daytime_votes_history = [IdentifiedModel(value=votes)]
    else:
        state.nighttime_votes_history = [IdentifiedModel(value=votes)]
    # execution
    actual = _eliminate_player(state)
    # assert
    if timespan == ETimeSpan.day:
        assert actual['daytime_vote_result_history'][-1].value is None
    else:
        assert actual['nighttime_vote_result_history'][-1].value is None


@pytest.mark.parametrize(
    'timespan',
    [
        ETimeSpan.day,
        ETimeSpan.night,
    ],
)
def test__eliminate_player_with_tied_valid_votes(
    timespan: ETimeSpan,
) -> None:
    # preparation
    votes = {'player1': 'player2', 'player2': 'player3', 'player3': 'player1'}
    state = StateModel(alive_players_names=list(votes.keys()), timespan=timespan)  # noqa
    if timespan == ETimeSpan.day:
        state.daytime_votes_history = [IdentifiedModel(value=votes)]
    else:
        state.nighttime_votes_history = [IdentifiedModel(value=votes)]
    # execution
    actual = _eliminate_player(state, select_from_same_votes=lambda lst: sorted(lst)[0])  # noqa
    # assert
    if timespan == ETimeSpan.day:
        assert actual['daytime_vote_result_history'][-1].value is 'player1'  # noqa
    else:
        assert actual['nighttime_vote_result_history'][-1].value is 'player1'  # noqa


@pytest.mark.parametrize(
    'timespan',
    [
        ETimeSpan.day,
        ETimeSpan.night,
    ],
)
def test__eliminate_player_with_top_saved(
    timespan: ETimeSpan,
) -> None:
    # preparation
    safe_players = ['player3']
    votes = {'player1': 'player3', 'player2': 'player3', 'player3': 'player1'}  # noqa
    state = StateModel(
        alive_players_names=list(votes.keys()),
        timespan=timespan,
        safe_players_names=set(safe_players),
    )  # noqa
    if timespan == ETimeSpan.day:
        state.daytime_votes_history = [IdentifiedModel(value=votes)]
    else:
        state.nighttime_votes_history = [IdentifiedModel(value=votes)]
    # execution
    actual = _eliminate_player(state)
    # assert
    if timespan == ETimeSpan.day:
        assert actual['daytime_vote_result_history'][-1].value is 'player1'  # noqa
    else:
        assert actual['nighttime_vote_result_history'][-1].value is 'player1'  # noqa
