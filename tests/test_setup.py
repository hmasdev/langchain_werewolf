from functools import partial
import click
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import (
    DEFAULT_MODEL,
    CLI_PROMPT_COLOR,
    CLI_PROMPT_SUFFIX,
)
from langchain_werewolf.enums import EInputOutputType, ERole, ESystemOutputType  # noqa
from langchain_werewolf.game_players.base import BaseGamePlayer
from langchain_werewolf.models.config import PlayerConfig
from langchain_werewolf.setup import (
    _create_echo_runnable_by_player,
    _create_echo_runnable_by_system,
    _generate_base_runnable,
    create_echo_runnable,
    generate_players,
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
    player_config = PlayerConfig(model='cli', input_output_type=EInputOutputType.standard)  # noqa
    expected_args = []  # type: ignore
    expected_kwargs = {
        'input_func': player_config.input_output_type,
        'styler': partial(click.style, fg=CLI_PROMPT_COLOR),
        'prompt_suffix': CLI_PROMPT_SUFFIX,
    }
    create_input_runnable_mock = mocker.patch(
        'langchain_werewolf.setup.create_input_runnable',
        return_value=mocker.MagicMock(spec=Runnable[str, str]),
    )
    # execution
    _generate_base_runnable(player_config.model, player_config.input_output_type)  # noqa
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
        PlayerConfig(role=ERole.Werewolf, model='cli', input_output_type=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.Knight, model='cli', input_output_type=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.FortuneTeller, model='cli', input_output_type=EInputOutputType.standard),  # noqa
        PlayerConfig(role=ERole.Villager, model='cli', input_output_type=EInputOutputType.standard),  # noqa
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


def test__create_echo_runnable_by_player() -> None:
    # TODO: implement more detailed test
    player = BaseGamePlayer.instantiate(
        role=ERole.Villager,
        name='Alice',
        runnable=RunnableLambda(str),
    )
    _create_echo_runnable_by_player(
        player=player,
        player_config=None,
        model='cli',
        input_output_type=EInputOutputType.standard,
    )


@pytest.mark.parametrize(
    'level',
    [
        ESystemOutputType.off,
        ESystemOutputType.all,
        ESystemOutputType.public,
        'name',
    ],
)
def test__create_echo_runnable_by_system(
    level: ESystemOutputType | str,
) -> None:
    # TODO: implement more detailed test
    _create_echo_runnable_by_system(
        kind=EInputOutputType.standard,
        level=level,
        player_names=['name'],
        model='cli',
    )


def test__create_echo_runnable_by_system_with_invalid_level() -> None:
    # assert
    with pytest.raises(ValueError):
        _create_echo_runnable_by_system(
            kind=EInputOutputType.standard,
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
        input_output_type=EInputOutputType.standard,
        cli_output_level=ESystemOutputType.all,
        players=players,
        players_cfg=[PlayerConfig(role=player.role) for player in players],
    )
