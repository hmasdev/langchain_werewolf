from langchain_core.runnables import RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.enums import ERole
from langchain_werewolf.game.night_action import (
    _master_ask_player_to_act_in_night,
    _skip_player_act_in_night,
    _player_act_in_night,
    GeneratePromptInputForNightAction,
)
from langchain_werewolf.game_players.base import BaseGamePlayer
from langchain_werewolf.models.state import StateModel


def test__master_ask_player_to_act_in_night() -> None:
    # preparation
    state = StateModel(alive_players_names=['player1', 'player2'])
    player = BaseGamePlayer.instantiate(
        name='player',
        role=ERole.FortuneTeller,
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
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.message == generate_prompt(GeneratePromptInputForNightAction(role=player.role.value, night_action=player.night_action, question_to_decide_night_action=player.question_to_decide_night_action, alive_players_names=state.alive_players_names))  # type: ignore # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.participants == frozenset([player.name, GAME_MASTER_NAME])  # noqa


@pytest.mark.parametrize(
    'player',
    [
        BaseGamePlayer.instantiate(
            name='player',
            role=ERole.Werewolf,
            runnable=RunnableLambda(str),
        ),
        BaseGamePlayer.instantiate(
            name='player',
            role=ERole.Villager,
            runnable=RunnableLambda(str),
        ),
        BaseGamePlayer.instantiate(
            name='player',
            role=ERole.FortuneTeller,
            runnable=RunnableLambda(str),
        ),
        BaseGamePlayer.instantiate(
            name='player',
            role=ERole.Knight,
            runnable=RunnableLambda(str),
        ),
    ]
)
def test__player_act_in_night(
    player: BaseGamePlayer,
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
            ERole.Werewolf,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'skip'
        ),
        (
            'player3',
            ERole.Villager,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'skip'
        ),
        (
            'player1',
            ERole.Villager,
            ['player1', 'player2'],
            'not_skip',
            'skip',
            'not_skip'
        ),
    ]
)
def test__skip_player_act_in_night(
    name: str,
    role: ERole,
    alive_players: list[str],
    not_skip_destination_node_name: str,
    skip_destination_node_name: str,
    expected: str,
) -> None:
    # preparation
    state = StateModel(alive_players_names=alive_players)
    player = BaseGamePlayer.instantiate(
        name=name,
        role=role,
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
