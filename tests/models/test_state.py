from datetime import datetime as dt
from typing import Generator
import pytest
from langchain_werewolf.enums import EResult, ETimeSpan
from langchain_werewolf.models.general import IdentifiedModel
from langchain_werewolf.models.state import (
    ChatHistoryModel,
    MsgModel,
    StateModel,
    _get_specific_chat,
    _integrate_chat_histories,
    _reduce_chat_state,
    _reduce_votes_current,
    create_dict_to_add_safe_player,
    create_dict_to_reset_state,
    create_dict_to_update_alive_players,
    create_dict_to_update_day,
    create_dict_to_record_chat,
    create_dict_to_update_chat_remaining_number,
    create_dict_to_update_current_speaker,
    create_dict_to_update_daytime_vote_result_history,
    create_dict_to_update_daytime_votes_current,
    create_dict_to_update_daytime_votes_history,
    create_dict_to_update_nighttime_vote_result_history,
    create_dict_to_update_nighttime_votes_current,
    create_dict_to_update_nighttime_votes_history,
    create_dict_to_update_result,
    create_dict_to_update_timespan,
    create_dict_without_state_updated,
    get_related_chat_histories,
    get_related_messsages,
    get_related_messsages_with_id,
)


@pytest.fixture(scope='function')
def state_fixture() -> Generator[StateModel, None, None]:
    yield StateModel(
        alive_players_names=['Alice', 'Bob'],
        chat_state={
            frozenset({'Alice', 'Bob'}): ChatHistoryModel(
                names=frozenset({'Alice', 'Bob'}),
                messages=[
                    IdentifiedModel[MsgModel](
                        id='dummy',
                        value=MsgModel(
                            name='Alice',
                            timestamp=dt(2021, 1, 1),
                            message='hello, world',
                            participants=frozenset({'Alice', 'Bob'}),
                        ),
                    ),
                    IdentifiedModel[MsgModel](
                        id='dummy',
                        value=MsgModel(
                            name='Bob',
                            timestamp=dt(2021, 1, 3),
                            message='hello, world',
                            participants=frozenset({'Alice', 'Bob'}),
                        ),
                    ),
                ],
            ),
            frozenset({'Alice'}): ChatHistoryModel(
                names=frozenset({'Alice'}),
                messages=[
                    IdentifiedModel[MsgModel](
                        id='dummy',
                        value=MsgModel(
                            name='Alice',
                            timestamp=dt(2021, 1, 2),
                            message='self messsasge',
                            participants=frozenset({'Alice'}),
                        ),
                    ),
                ],
            ),
            frozenset({'Bob'}): ChatHistoryModel(
                names=frozenset({'Bob'}),
                messages=[
                    IdentifiedModel[MsgModel](
                        id='dummy',
                        value=MsgModel(
                            name='Bob',
                            timestamp=dt(2021, 1, 3),
                            message='hello, Alice',
                            participants=frozenset({'Bob'}),
                        ),
                    ),
                ],
            ),
        },
    )


def test_MsgModel_format() -> None:
    # preparation
    msg = MsgModel(
        name='Alice',
        timestamp=dt(2021, 1, 1),
        message='hello, world',
        participants=frozenset({'Alice', 'Bob'}),
        template='{name}{timestamp}{participants}{message}',
    )
    expected = 'Alice2021-01-01 00:00:00.000000[\'Alice\', \'Bob\']hello, world'  # noqa
    # execution
    actual = msg.format()
    # assert
    assert actual == expected


def test_MsgModel_serialize_timestamp() -> None:
    # preparation
    msg = MsgModel(
        name='Alice',
        timestamp=dt(2021, 1, 1),
        message='hello, world',
        participants=frozenset({'Alice', 'Bob'}),
    )
    expected = '2021-01-01 00:00:00.000000'
    # execution
    actual = msg.serialize_timestamp(msg.timestamp)
    # assert
    assert actual == expected


def test_MsgModel_serialize_participants() -> None:
    # preparation
    msg = MsgModel(
        name='Alice',
        timestamp=dt(2021, 1, 1),
        message='hello, world',
        participants=frozenset({'Alice', 'Bob'}),
    )
    expected = ['Alice', 'Bob']
    # execution
    actual = msg.serialize_participants(msg.participants)
    # assert
    assert actual == expected


def test_ChatHistoryModel_serialize_names() -> None:
    # preparation
    chat_history = ChatHistoryModel(
        names=frozenset({'Alice', 'Bob'}),
        messages=[]
    )
    expected = ['Alice', 'Bob']
    # execution
    actual = chat_history.serialize_names(chat_history.names)
    # assert
    assert actual == expected


def test_ChatHistoryModel_preprocess_messages() -> None:
    # preparation
    messages: list[MsgModel | IdentifiedModel[MsgModel]] = [
        IdentifiedModel[MsgModel](
            id='dummy',
            value=MsgModel(
                name='Alice',
                timestamp=dt(2021, 1, 1),
                message='hello, world',
                participants=frozenset({'Alice', 'Bob'}),
            ),
        ),
        MsgModel(
            name='Bob',
            timestamp=dt(2021, 1, 3),
            message='hello, world',
            participants=frozenset({'Alice', 'Bob'}),
        ),
    ]
    # execution
    actual = ChatHistoryModel.preprocess_messages(messages)
    # assert
    assert actual[0] == messages[0]
    assert actual[1].id
    assert actual[1].value == messages[1]


def test_create_dict_to_record_chat() -> None:
    # preparation
    sender = 'Alice'
    participants = ['Alice', 'Bob']
    message = 'hello, world'
    # execution
    actual = create_dict_to_record_chat(sender, participants, message)
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][frozenset(participants)]
    assert actual['chat_state'][frozenset(participants)].names == frozenset(participants)  # noqa
    assert actual['chat_state'][frozenset(participants)].messages[0].value.name == sender  # noqa
    assert actual['chat_state'][frozenset(participants)].messages[0].value.message == message  # noqa


def test__reduce_chat_state() -> None:
    # preparation
    previous_chat_state = {
        frozenset({'Alice', 'Bob'}): ChatHistoryModel(
            names=frozenset({'Alice', 'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='0',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='1',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        frozenset({'Alice'}): ChatHistoryModel(
            names=frozenset({'Alice'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='2',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 2),
                        message='self messsasge',
                        participants=frozenset({'Alice'}),
                    ),
                ),
            ],
        ),
    }
    new_chat_state = {
        frozenset({'Alice', 'Bob'}): ChatHistoryModel(
            names=frozenset({'Alice', 'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='3',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 4),
                        message='new comment',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        frozenset({'Bob'}): ChatHistoryModel(
            names=frozenset({'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='4',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 4),
                        message='new comment',
                        participants=frozenset({'Bob'}),
                    ),
                ),
            ],
        ),
    }
    expected = {
        frozenset({'Alice', 'Bob'}): ChatHistoryModel(
            names=frozenset({'Alice', 'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='0',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='1',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='3',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 4),
                        message='new comment',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        frozenset({'Alice'}): ChatHistoryModel(
            names=frozenset({'Alice'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='2',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 2),
                        message='self messsasge',
                        participants=frozenset({'Alice'}),
                    ),
                ),
            ],
        ),
        frozenset({'Bob'}): ChatHistoryModel(
            names=frozenset({'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='4',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 4),
                        message='new comment',
                        participants=frozenset({'Bob'}),
                    ),
                ),
            ],
        ),
    }
    # execution
    actual = _reduce_chat_state(previous_chat_state, new_chat_state)
    # assert
    assert actual == expected


def test_create_dict_to_reset_temporal_state(
    state_fixture: StateModel,
) -> None:
    # preparation
    expected: dict[str, object] = {
        'safe_players_names': set(),
        'current_speaker': None,
        'n_chat_remaining': None,
        'daytime_votes_current': {},
        'nighttime_votes_current': {},
    }
    # execution
    actual = create_dict_to_reset_state(state_fixture)
    # assert
    assert actual == expected


def test_StateModel_serialize_safe_players_names(state_fixture: StateModel) -> None:  # noqa
    # preparation
    state_fixture.safe_players_names = {'Bob', 'Alice'}
    expected = ['Alice', 'Bob']
    # execution
    actual = state_fixture.serialize_safe_players_names(state_fixture.safe_players_names)  # noqa
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'result, expected',
    [
        (EResult.VillagersWin, EResult.VillagersWin.value),
        (EResult.WerewolvesWin, EResult.WerewolvesWin.value),
        (None, 'None'),
    ]
)
def test_StateModel_serialize_result(
    result: EResult | None,
    expected: str,
    state_fixture: StateModel,
) -> None:
    # execution
    actual = state_fixture.serialize_result(result)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'previous_votes,new_votes,expected',
    [
        (
            # Normal case: Alice votes Bob
            {},
            {'Alice': 'Bob'},
            {'Alice': 'Bob'},
        ),
        (
            # Normal case: Alice votes Bob with previous votes are None
            None,
            {'Alice': 'Bob'},
            {'Alice': 'Bob'},
        ),
        (
            # Normal case: Alice changes her vote
            {'Alice': 'Bob'},
            {'Alice': 'Charlie'},
            {'Alice': 'Charlie'},
        ),
        (
            # Normal case: Alice votes Bob, Charlie newly votes Bob
            {'Alice': 'Bob'},
            {'Charlie': 'Bob'},
            {'Alice': 'Bob', 'Charlie': 'Bob'},
        ),
        (
            # Abnormal case: new_votes is None
            {'Alice': 'Bob'},
            None,
            {'Alice': 'Bob'},
        ),
        (
            # Reset case
            {'Alice': 'Bob'},
            {},
            {},
        ),
    ]
)
def test__reduce_votes_current(
    previous_votes: dict[str, str] | None,
    new_votes: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    # execution
    actual = _reduce_votes_current(previous_votes, new_votes)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'day',
    [1, 2],
)
def test_create_dict_to_update_day(day: int) -> None:
    # execution
    actual = create_dict_to_update_day(day)
    # assert
    assert actual == {'day': day}


@pytest.mark.parametrize(
    'timespan',
    [ETimeSpan.day, ETimeSpan.night],
)
def test_create_dict_to_update_timespan(timespan: ETimeSpan) -> None:
    # execution
    actual = create_dict_to_update_timespan(timespan)
    # assert
    assert actual == {'timespan': timespan}


def test_create_dict_to_add_safe_player() -> None:
    # preparation
    name = 'Alice'
    # execution
    actual = create_dict_to_add_safe_player(name)
    # assert
    assert actual == {'safe_players_names': {name}}


def test_create_dict_to_update_current_speaker() -> None:
    # preparation
    name = 'Alice'
    # execution
    actual = create_dict_to_update_current_speaker(name)
    # assert
    assert actual == {'current_speaker': name}


def test_create_dict_to_update_chat_remaining_number() -> None:
    # preparation
    n_chat_remaining = 3
    # execution
    actual = create_dict_to_update_chat_remaining_number(n_chat_remaining)
    # assert
    assert actual == {'n_chat_remaining': n_chat_remaining}


def test_create_dict_to_update_daytime_vote_result_history() -> None:
    # preparation
    result = 'x'
    # execution
    actual = create_dict_to_update_daytime_vote_result_history(result)
    # assert
    assert actual['daytime_vote_result_history']
    assert actual['daytime_vote_result_history'][0].id
    assert actual['daytime_vote_result_history'][0].value == result


def test_create_dict_to_update_daytime_votes_current() -> None:
    # preparation
    votes = {'Alice': 'Bob'}
    # execution
    actual = create_dict_to_update_daytime_votes_current(votes)
    # assert
    assert actual == {'daytime_votes_current': votes}


def test_create_dict_to_update_nighttime_vote_result_history() -> None:
    # preparation
    result = 'x'
    # execution
    actual = create_dict_to_update_nighttime_vote_result_history(result)
    # assert
    assert actual['nighttime_vote_result_history']
    assert actual['nighttime_vote_result_history'][0].id
    assert actual['nighttime_vote_result_history'][0].value == result


def test_create_dict_to_update_nighttime_votes_current() -> None:
    # preparation
    votes = {'Alice': 'Bob'}
    # execution
    actual = create_dict_to_update_nighttime_votes_current(votes)
    # assert
    assert actual == {'nighttime_votes_current': votes}


def test_create_dict_to_update_daytime_votes_history() -> None:
    # preparation
    votes_history = {
        'Alice': 'Bob',
        'Bob': 'Charlie',
    }
    # execution
    actual = create_dict_to_update_daytime_votes_history(votes_history)
    # assert
    assert actual == {'daytime_votes_history': [votes_history]}


def test_create_dict_to_update_nighttime_votes_history() -> None:
    # preparation
    votes_history = {
        'Alice': 'Bob',
        'Bob': 'Charlie',
    }
    # execution
    actual = create_dict_to_update_nighttime_votes_history(votes_history)
    # assert
    assert actual == {'nighttime_votes_history': [votes_history]}


def test_create_dict_to_update_alive_players() -> None:
    # preparation
    alive_players_names = ['Alice', 'Bob']
    # execution
    actual = create_dict_to_update_alive_players(alive_players_names)
    # assert
    assert actual == {'alive_players_names': alive_players_names}


@pytest.mark.parametrize(
    'result',
    [
        EResult.VillagersWin,
        EResult.WerewolvesWin,
        None,
    ],
)
def test_create_dict_to_update_result(
    result: EResult | None,
) -> None:
    # execution
    actual = create_dict_to_update_result(result)
    # assert
    assert actual == {'result': result}


def test_create_dict_without_state_updated() -> None:
    # assert
    assert create_dict_without_state_updated(StateModel(alive_players_names=['dummy'])) == {'chat_state': {}}  # noqa


def test_get_related_chat_histories(state_fixture: StateModel) -> None:
    # preparation
    name: str = 'Alice'
    state = state_fixture
    expected = {
        frozenset({'Alice', 'Bob'}): ChatHistoryModel(
            names=frozenset({'Alice', 'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        frozenset({'Alice'}): ChatHistoryModel(
            names=frozenset({'Alice'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 2),
                        message='self messsasge',
                        participants=frozenset({'Alice'}),
                    ),
                ),
            ],
        ),
    }
    # execution
    actual = get_related_chat_histories(name, state)
    # assert
    assert actual == expected


def test__integrate_chat_histories(state_fixture: StateModel) -> None:
    # preparation
    chat_histories = [
        ChatHistoryModel(
            names=frozenset({'Alice', 'Bob'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        ChatHistoryModel(
            names=frozenset({'Alice'}),
            messages=[
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 2),
                        message='self messsasge',
                        participants=frozenset({'Alice'}),
                    ),
                ),
            ],
        ),
    ]
    expected = [
        IdentifiedModel[MsgModel](
            id='dummy',
            value=MsgModel(
                name='Alice',
                timestamp=dt(2021, 1, 1),
                message='hello, world',
                participants=frozenset({'Alice', 'Bob'}),
            ),
        ),
        IdentifiedModel[MsgModel](
            id='dummy',
            value=MsgModel(
                name='Alice',
                timestamp=dt(2021, 1, 2),
                message='self messsasge',
                participants=frozenset({'Alice'}),
            ),
        ),
        IdentifiedModel[MsgModel](
            id='dummy',
            value=MsgModel(
                name='Bob',
                timestamp=dt(2021, 1, 3),
                message='hello, world',
                participants=frozenset({'Alice', 'Bob'}),
            ),
        ),
    ]
    # execution
    actual = _integrate_chat_histories(*chat_histories)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'names',
    [
        frozenset({'Alice', 'Bob'}),
        frozenset({'Alice'}),
        frozenset({'Bob'}),
    ],
)
def test__get_specific_chat(
    names: frozenset[str],
    state_fixture: StateModel,
) -> None:
    # preparation
    state = state_fixture
    expected = state.chat_state[names]
    # execution
    actual = _get_specific_chat(names, state)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'name, expected',
    [
        (
            'Alice',
            [
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 2),
                        message='self messsasge',
                        participants=frozenset({'Alice'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
        (
            ['Alice', 'Bob'],
            [
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Alice',
                        timestamp=dt(2021, 1, 1),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
                IdentifiedModel[MsgModel](
                    id='dummy',
                    value=MsgModel(
                        name='Bob',
                        timestamp=dt(2021, 1, 3),
                        message='hello, world',
                        participants=frozenset({'Alice', 'Bob'}),
                    ),
                ),
            ],
        ),
    ]
)
def test_get_related_messages_with_id(
    name: str | list[str],
    expected: list[IdentifiedModel[MsgModel]],
    state_fixture: StateModel,
) -> None:
    # preparation
    state = state_fixture
    # execution
    actual = get_related_messsages_with_id(name, state)
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    'name, expected',
    [
        (
            'Alice',
            [
                MsgModel(
                    name='Alice',
                    timestamp=dt(2021, 1, 1),
                    message='hello, world',
                    participants=frozenset({'Alice', 'Bob'}),
                ),
                MsgModel(
                    name='Alice',
                    timestamp=dt(2021, 1, 2),
                    message='self messsasge',
                    participants=frozenset({'Alice'}),
                ),
                MsgModel(
                    name='Bob',
                    timestamp=dt(2021, 1, 3),
                    message='hello, world',
                    participants=frozenset({'Alice', 'Bob'}),
                ),
            ],
        ),
        (
            ['Alice', 'Bob'],
            [
                MsgModel(
                    name='Alice',
                    timestamp=dt(2021, 1, 1),
                    message='hello, world',
                    participants=frozenset({'Alice', 'Bob'}),
                ),
                MsgModel(
                    name='Bob',
                    timestamp=dt(2021, 1, 3),
                    message='hello, world',
                    participants=frozenset({'Alice', 'Bob'}),
                ),
            ],
        ),
    ]
)
def test_get_related_messages(
    name: str | list[str],
    expected: list[IdentifiedModel[MsgModel]],
    state_fixture: StateModel,
) -> None:
    # preparation
    state = state_fixture
    # execution
    actual = get_related_messsages(name, state)
    # assert
    assert actual == expected
