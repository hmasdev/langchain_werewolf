from langchain_core.runnables import RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game.night_action import (
    _master_ask_player_to_act_in_night,
    _skip_player_act_in_night,
    _player_act_in_night,
    GeneratePromptInputForNightAction,
)
from langchain_werewolf.game_players import BaseGamePlayerRole
from langchain_werewolf.game_players.player_roles import (
    FortuneTeller,
    Knight,
    Villager,
    Werewolf,
)
from langchain_werewolf.game_players.registry import PlayerRoleRegistry
from langchain_werewolf.models.state import StateModel


def test__master_ask_player_to_act_in_night() -> None:
    # preparation
    state = StateModel(alive_players_names=['player1', 'player2'])
    player = PlayerRoleRegistry.create_player(
        name='player',
        key=FortuneTeller.role,
        runnable=RunnableLambda(str),
    )

    def generate_prompt(m: GeneratePromptInputForNightAction) -> str:
        return f'{m.role}{m.night_action}{m.question_to_decide_night_action}{m.alive_players_names}'  # noqa

    # execution
    actual = _master_ask_player_to_act_in_night(
        state,
        player,
        lambda m: f'{m.role}{m.night_action}{m.question_to_decide_night_action}{m.alive_players_names}',  # noqa
    )
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])]
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages  # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.name == GAME_MASTER_NAME  # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.message == generate_prompt(GeneratePromptInputForNightAction(role=player.role, night_action=player.night_action, question_to_decide_night_action=player.question_to_decide_night_action, alive_players_names=state.alive_players_names))  # type: ignore # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.participants == frozenset([player.name, GAME_MASTER_NAME])  # noqa


@pytest.mark.parametrize(
    'player',
    [
        PlayerRoleRegistry.create_player(
            key=Werewolf.role,
            name='player',
            runnable=RunnableLambda(str),
        ),
        PlayerRoleRegistry.create_player(
            key=Villager.role,
            name='player',
            runnable=RunnableLambda(str),
        ),
        PlayerRoleRegistry.create_player(
            key=FortuneTeller.role,
            name='player',
            runnable=RunnableLambda(str),
        ),
        PlayerRoleRegistry.create_player(
            key=Knight.role,
            name='player',
            runnable=RunnableLambda(str),
        ),
    ]
)
def test__player_act_in_night(
    player: BaseGamePlayerRole,
    mocker: MockerFixture,
) -> None:
    # preparation
    expected = {'dummy_for_test': 'dummy_for_test'}
    state = StateModel(alive_players_names=['player1', 'player2'])
    players = [player]
    object.__setattr__(player, 'act_in_night', mocker.MagicMock(return_value=expected))  # noqa
    # execution
    actual = _player_act_in_night(state, player, players)
    # assert
    for k, v in expected.items():
        assert actual[k] == v


@pytest.mark.parametrize(
    'name, role, alive_players, not_skip_destination_node_name, skip_destination_node_name, expected',  # noqa
    [
        (
            'player1',
            Werewolf.role,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'skip'
        ),
        (
            'player3',
            Villager.role,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'skip'
        ),
        (
            'player1',
            Villager.role,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'not_skip'
        ),
    ]
)
def test__skip_player_act_in_night(
    name: str,
    role: str,
    alive_players: list[str],
    not_skip_destination_node_name: str,
    skip_destination_node_name: str,
    expected: str,
) -> None:
    # preparation
    state = StateModel(alive_players_names=alive_players)
    player = PlayerRoleRegistry.create_player(
        name=name,
        key=role,
        runnable=RunnableLambda(str),
    )
    # execution
    actual = _skip_player_act_in_night(
        state,
        player,
        not_skip_destination_node_name,
        skip_destination_node_name,
    )
    # assert
    assert actual == expected
