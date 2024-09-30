from langchain_core.runnables import RunnableLambda
import pytest
from langchain_werewolf.enums import ERole, EResult
from langchain_werewolf.game.check_result import (
    check_victory_condition,
)
from langchain_werewolf.game_players.base import BaseGamePlayer
from langchain_werewolf.models.state import StateModel


@pytest.mark.parametrize(
    'n_alive_players, n_werewolves, expected',
    [
        (
            10,
            6,
            EResult.WerewolvesWin,
        ),
        (
            10,
            5,
            EResult.WerewolvesWin,
        ),
        (
            10,
            4,
            None,
        ),
        (
            10,
            1,
            None,
        ),
        (
            10,
            0,
            EResult.VillagersWin,
        ),
    ]
)
def test_check_victory_condition(
    n_alive_players: int,
    n_werewolves: int,
    expected: dict[str, EResult | None],
) -> None:
    # preparation
    players = [
        BaseGamePlayer.instantiate(
            name=f'player{i}',
            role=ERole.Werewolf if i < n_werewolves else ERole.Villager,
            runnable=RunnableLambda(str),
        )
        for i in range(n_alive_players)
    ]
    state = StateModel(alive_players_names=[p.name for p in players])  # noqa
    # execution
    actual = check_victory_condition(state, players)
    # assert
    assert actual == {'result': expected}
