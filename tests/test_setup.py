from functools import partial
from typing import Callable
from unittest import mock
import click
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import (
    BASE_LANGUAGE,
    DEFAULT_MODEL,
    CLI_PROMPT_COLOR,
    CLI_PROMPT_SUFFIX,
    GAME_MASTER_NAME,
)
from langchain_werewolf.enums import (
    EInputOutputType,
    ERole,
    ESystemOutputType,
    ETimeSpan,
)
from langchain_werewolf.game_players.base import BaseGamePlayer
from langchain_werewolf.models.config import PlayerConfig
from langchain_werewolf.models.state import (
    ChatHistoryModel,
    IdentifiedModel,
    MsgModel,
    StateModel,
)
from langchain_werewolf.setup import (
    _create_echo_runnable_by_player,
    _create_echo_runnable_by_system,
    _generate_base_runnable,
    create_echo_runnable,
    generate_players,
)


PLAYER_NAMES4TEST = ('Alice', 'Bob', 'Charley')
STATE4TEST = StateModel(
    day=1,
    timespan=ETimeSpan.day,
    result=None,
    chat_state={
        frozenset({GAME_MASTER_NAME, name}): ChatHistoryModel(
            names=frozenset({GAME_MASTER_NAME, name}),
            messages=[
                IdentifiedModel[MsgModel](value=MsgModel(
                    name=GAME_MASTER_NAME,
                    message=f'Hello, {name}!',
                    participants=frozenset({GAME_MASTER_NAME, name}),
                )),
                IdentifiedModel[MsgModel](value=MsgModel(
                    name=name,
                    message=f'Hello, {GAME_MASTER_NAME}!',
                    participants=frozenset({GAME_MASTER_NAME, name}),
                )),
                IdentifiedModel[MsgModel](value=MsgModel(
                    name=GAME_MASTER_NAME,
                    message='How are you?',
                    participants=frozenset({GAME_MASTER_NAME, name}),
                )),
            ],
        )
        for name in PLAYER_NAMES4TEST
    } | {
        frozenset(set(PLAYER_NAMES4TEST) | {GAME_MASTER_NAME}): ChatHistoryModel(  # noqa
            names=frozenset(set(PLAYER_NAMES4TEST) | {GAME_MASTER_NAME}),
            messages=[
                IdentifiedModel[MsgModel](value=MsgModel(
                    name=GAME_MASTER_NAME,
                    message=f'Hello, {", ".join(PLAYER_NAMES4TEST)}!',
                    participants=frozenset(set(PLAYER_NAMES4TEST) | {GAME_MASTER_NAME}),  # noqa
                )),
            ] + [
                IdentifiedModel[MsgModel](value=MsgModel(
                    name=name,
                    message=f'Hello, everyone!',
                    participants=frozenset(set(PLAYER_NAMES4TEST) | {GAME_MASTER_NAME}),  # noqa
                ))
                for name in PLAYER_NAMES4TEST
            ],
        )
    },
    alive_players_names=PLAYER_NAMES4TEST,
    safe_players_names=[],
    current_speaker=None,
    n_chat_remaining=0,
    daytime_vote_result_history=[],
    nighttime_vote_result_history=[],
    daytime_votes_current={},
    nighttime_votes_current={},
    daytime_votes_history=[],
    nighttime_votes_history=[],
)


@pytest.mark.parametrize(
    'seed',
    [-1, 0, 1],
)
def test__generate_base_runnable_with_model_is_none(
    seed: int,
    mocker: MockerFixture,
) -> None:
    # preparation
    expected_args = [DEFAULT_MODEL]
    expected_kwargs = {'seed': seed if seed >= 0 else None}
    create_chat_model_mock = mocker.patch(
        'langchain_werewolf.setup.create_chat_model',
        return_value=mocker.MagicMock(spec=BaseChatModel),
    )
    # execution
    _generate_base_runnable(None, None, seed=seed)
    # assert
    create_chat_model_mock.assert_called_once_with(*expected_args, **expected_kwargs)  # noqa


@pytest.mark.parametrize(
    'model, seed',
    [
        ('gpt-3.5-turbo', -1),
        ('gpt-4o-mini', 0),
        ('gpt-4-turbo', 1),
    ],
)
def test__generate_base_runnable_with_chat_model(
    model: str,
    seed: int,
    mocker: MockerFixture,
) -> None:
    # preparation
    expected_args = [model]
    expected_kwargs = {'seed': seed if seed >= 0 else None}
    create_chat_model_mock = mocker.patch(
        'langchain_werewolf.setup.create_chat_model',
        return_value=mocker.MagicMock(spec=BaseChatModel),
    )
    # execution
    _generate_base_runnable(model, None, seed=seed)
    # assert
    create_chat_model_mock.assert_called_once_with(*expected_args, **expected_kwargs)  # noqa


def test__generate_base_runnable_with_cli_player_config(
    mocker: MockerFixture,
) -> None:
    # preparation
    player_config = PlayerConfig(model='cli', player_input_interface=EInputOutputType.standard)  # noqa
    expected_args = []  # type: ignore
    expected_kwargs = {
        'input_func': player_config.player_input_interface,
        'styler': partial(click.style, fg=CLI_PROMPT_COLOR),
        'prompt_suffix': CLI_PROMPT_SUFFIX,
    }
    create_input_runnable_mock = mocker.patch(
        'langchain_werewolf.setup.create_input_runnable',
        return_value=mocker.MagicMock(spec=Runnable[str, str]),
    )
    # execution
    _generate_base_runnable(player_config.model, player_config.player_input_interface)  # noqa
    # assert
    create_input_runnable_mock.assert_called_once()
    actual_args, actual_kwargs = create_input_runnable_mock.call_args
    assert actual_args == tuple(expected_args)
    assert actual_kwargs['input_func'] == expected_kwargs['input_func']
    assert actual_kwargs['prompt_suffix'] == expected_kwargs['prompt_suffix']
    assert all([
        actual_kwargs['styler'].func == expected_kwargs['styler'].func,  # type: ignore # noqa
        actual_kwargs['styler'].args == expected_kwargs['styler'].args,  # type: ignore # noqa
        actual_kwargs['styler'].keywords == expected_kwargs['styler'].keywords,  # type: ignore # noqa
    ])
    # NOTE: partial object is not equal to another partial object with the same function and arguments  # noqa


def test__generate_base_runnable_with_unsupported_config() -> None:
    # execution
    with pytest.raises(ValueError):
        _generate_base_runnable('cli', None)


@pytest.mark.parametrize(
    'n_players, n_werewolves, n_knights, n_fortune_tellers',
    [
        (1, 2, 0, 0),
        (1, 0, 2, 0),
        (1, 0, 0, 2),
        (2, 1, 1, 1),
    ],
)
def test_generate_players_with_invalid_n_xxxxxxs(
    n_players: int,
    n_werewolves: int,
    n_knights: int,
    n_fortune_tellers: int,
) -> None:
    # assert
    with pytest.raises(ValueError):
        generate_players(
            n_players,
            n_werewolves,
            n_knights,
            n_fortune_tellers,
            seed=0,
            custom_players=[],
        )


@pytest.mark.parametrize(
    'n_players, n_werewolves, n_knights, n_fortune_tellers, custom_players',
    [
        (5, 2, 0, 0, [PlayerConfig(role=ERole.Werewolf)]*3),
        (5, 0, 2, 0, [PlayerConfig(role=ERole.Knight)]*3),
        (5, 0, 0, 2, [PlayerConfig(role=ERole.FortuneTeller)]*3),
        (5, 1, 1, 1, [PlayerConfig(role=ERole.Villager)]*3),
        (2, 0, 0, 0, [PlayerConfig(role=ERole.Villager)]*3),
    ],
)
def test_generate_players_with_invalid_n_xxxxxxs_for_custom_players(
    n_players: int,
    n_werewolves: int,
    n_knights: int,
    n_fortune_tellers: int,
    custom_players: list[PlayerConfig],
) -> None:
    # assert
    with pytest.raises(ValueError):
        generate_players(
            n_players,
            n_werewolves,
            n_knights,
            n_fortune_tellers,
            seed=0,
            custom_players=custom_players,
        )


def test_generate_players() -> None:
    n_players = 8
    n_werewolves = 2
    n_knights = 2
    n_fortune_tellers = 2
    custom_players = [
        PlayerConfig(role=ERole.Werewolf, model='cli', player_input_interface=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.Knight, model='cli', player_input_interface=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.FortuneTeller, model='cli', player_input_interface=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.Villager, model='cli', player_input_interface=EInputOutputType.standard),  # noqa
    ]
    actual = generate_players(
        n_players,
        n_werewolves,
        n_knights,
        n_fortune_tellers,
        model='cli',
        seed=0,
        input_output_type=EInputOutputType.standard,
        custom_players=custom_players,
    )
    assert len(actual) == n_players
    assert sum([player.role == ERole.Werewolf for player in actual]) == n_werewolves  # noqa
    assert sum([player.role == ERole.Knight for player in actual]) == n_knights  # noqa
    assert sum([player.role == ERole.FortuneTeller for player in actual]) == n_fortune_tellers  # noqa
    assert sum([player.role == ERole.Villager for player in actual]) == n_players - n_werewolves - n_knights - n_fortune_tellers  # noqa


@pytest.mark.parametrize(
    'player_name, formatter',
    [
        (player_name, formatter)
        for player_name in PLAYER_NAMES4TEST
        for formatter in [
            None,
            '{timestamp}, {name}, {message}',
            lambda msg: f"{msg.timestamp}-{msg.name}-{msg.message}",  # noqa
        ]
    ],
)
def test__create_echo_runnable_by_player_whether_invoke_method_calls_formatter_and_output_prroperly_without_colors_and_language_translation(  # noqa
    player_name: str,
    formatter: Callable[[MsgModel], str] | str | None,
    mocker: MockerFixture,
) -> None:
    # preparation
    player_name: str = PLAYER_NAMES4TEST[0]
    formatter = MsgModel.format
    output_mock = mocker.Mock()
    player = BaseGamePlayer.instantiate(
        role=ERole.Villager,
        name=player_name,
        runnable=RunnableLambda(str),
        output=RunnableLambda(output_mock),
        formatter=formatter,
    )
    expected = sorted([
        mocker.call(formatter(msg.value))
        for k, chat_history in STATE4TEST.chat_state.items()
        if player_name in k
        for msg in chat_history.messages
    ], key=lambda msg: msg.timestamp)
    # execute
    echo_runnable = _create_echo_runnable_by_player(
        player=player,
        player_config=None,
        color=None,
        language=BASE_LANGUAGE,
    )
    echo_runnable.invoke(STATE4TEST)
    # assert
    output_mock.assert_has_calls(expected)


@pytest.mark.parametrize(
    'level, formatter, expected_messages',
    [
        (level, formatter, expected_messages)
        for level, expected_messages in zip(
                [
                ESystemOutputType.off,
                ESystemOutputType.all,
                ESystemOutputType.public,
                PLAYER_NAMES4TEST[0],
            ],
            [
                [],
                sorted([
                    msg.value
                    for k, chat_history in STATE4TEST.chat_state.items()
                    if GAME_MASTER_NAME in k
                    for msg in chat_history.messages
                ], key=lambda msg: msg.timestamp),
                sorted([
                    msg.value
                    for k, chat_history in STATE4TEST.chat_state.items()
                    if k == frozenset({GAME_MASTER_NAME} | set(PLAYER_NAMES4TEST))
                    for msg in chat_history.messages
                ], key=lambda msg: msg.timestamp),
                sorted([
                    msg.value
                    for k, chat_history in STATE4TEST.chat_state.items()
                    if PLAYER_NAMES4TEST[0] in k
                    for msg in chat_history.messages
                ], key=lambda msg: msg.timestamp),
            ]
        )
        for formatter in [
            None,
            '{timestamp}, {name}, {message}',
            lambda msg: f"{msg.timestamp}-{msg.name}-{msg.message}",  # noqa
        ]
    ],
)
def test__create_echo_runnable_by_system_whether_invoke_method_calls_formatter_and_output_prroperly_without_colors_and_language_translation(  # noqa
    level: ESystemOutputType | str,
    formatter: Callable[[MsgModel], str] | str | None,
    expected_messages: list[MsgModel],
    mocker: MockerFixture,
) -> None:
    # expected
    expected_formatter_calls: list[mock._Call]
    expected_output_func_calls: list[mock._Call]

    # preparation
    output_func_mock = mocker.Mock()
    if formatter is None:
        # create expected
        expected_formatter_calls = [mocker.call(msg) for msg in expected_messages]  # noqa
        expected_output_func_calls = [mocker.call(MsgModel.format(msg)) for msg in expected_messages]  # noqa
        # create spy
        formatter_spy = mocker.Mock(side_effect=MsgModel.format)
        mocker.patch('langchain_werewolf.setup.MsgModel.format', formatter_spy)  # noqa
    elif isinstance(formatter, str):
        # create expected
        expected_formatter_calls = [mocker.call(**msg.model_dump()) for msg in expected_messages]  # noqa
        expected_output_func_calls = [mocker.call(formatter.format(**msg.model_dump())) for msg in expected_messages]  # noqa
        # create spy
        formatter_spy = mocker.Mock(side_effect=formatter.format)
        formatter = mocker.Mock(spec=str)
        formatter.format = formatter_spy
    elif callable(formatter):
        # create expected
        expected_formatter_calls = [mocker.call(msg) for msg in expected_messages]  # noqa
        expected_output_func_calls = [mocker.call(formatter(msg)) for msg in expected_messages]  # noqa
        # create spy
        formatter_spy = mocker.Mock(side_effect=formatter)
        formatter = formatter_spy

    # check expected
    assert len(expected_messages) == len(expected_formatter_calls)
    assert len(expected_messages) == len(expected_output_func_calls)

    # execute
    echo_runnable = _create_echo_runnable_by_system(
        output_func=output_func_mock,
        level=level,
        player_names=list(PLAYER_NAMES4TEST),
        color=None,
        language=BASE_LANGUAGE,
        formatter=formatter,
    )
    echo_runnable.invoke(STATE4TEST)

    # assert
    formatter_spy.assert_has_calls(expected_formatter_calls)
    output_func_mock.assert_has_calls(expected_output_func_calls)


def test__create_echo_runnable_by_system_with_invalid_level() -> None:
    # assert
    with pytest.raises(ValueError):
        _create_echo_runnable_by_system(
            output_func=EInputOutputType.standard,
            level='invalid',
            player_names=['name'],
        )


def test_create_echo_runnable(mocker: MockerFixture) -> None:
    # TODO: implement more detailed test
    # preparation
    mocker.patch(
        'langchain_werewolf.setup.create_chat_model',
        return_value=mocker.MagicMock(spec=BaseChatModel),
    )
    players = [
        BaseGamePlayer.instantiate(
            role=ERole.Villager,
            name='Alice',
            runnable=RunnableLambda(str),
        ),
        BaseGamePlayer.instantiate(
            role=ERole.Werewolf,
            name='Bob',
            runnable=RunnableLambda(str),
        ),
        BaseGamePlayer.instantiate(
            role=ERole.Knight,
            name='Charley',
            runnable=RunnableLambda(str),
        ),
    ]
    # execute
    create_echo_runnable(
        system_output_interface=EInputOutputType.standard,
        system_output_level=ESystemOutputType.all,
        players=players,
        players_cfg=[PlayerConfig(role=player.role) for player in players],
    )
