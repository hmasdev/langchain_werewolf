import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
import pytest
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game_players import BaseGamePlayer
from langchain_werewolf.game_players.player_roles import (
    FortuneTeller,
    Knight,
    Villager,
    Werewolf,
)
from langchain_werewolf.models.state import StateModel

load_dotenv()


@pytest.mark.parametrize(
    'players, runnable_output, expected',
    [
        (
            [
                Villager(name='Alice', runnable=RunnableLambda(str)),
                Werewolf(name='Bob', runnable=RunnableLambda(str)),
            ],
            'Alice',
            'Alice is not a werewolf',
        ),
        (
            [
                Villager(name='Alice', runnable=RunnableLambda(str)),
                Werewolf(name='Bob', runnable=RunnableLambda(str)),
            ],
            'Bob',
            'Bob is a werewolf',
        ),
        (
            [
                FortuneTeller(name='Alice', runnable=RunnableLambda(str)),
                Werewolf(name='Bob', runnable=RunnableLambda(str)),
            ],
            'Alice',
            'Alice is not a werewolf',
        ),
        (
            [
                Knight(name='Alice', runnable=RunnableLambda(str)),
                Werewolf(name='Bob', runnable=RunnableLambda(str)),
            ],
            'Alice',
            'Alice is not a werewolf',
        ),
    ]
)
def test_fortune_teller_act_in_night(
    players: list[BaseGamePlayer],
    runnable_output: str,
    expected: dict[str, object],
) -> None:
    # preparation
    player = FortuneTeller(
        name='Charlie',
        runnable=RunnableLambda(lambda _: runnable_output).with_types(output_type=str),  # noqa
    )
    # execution
    actual = player.act_in_night(
        players,
        [],
        StateModel(alive_players_names=[player.name]+[p.name for p in players]),  # noqa
    )
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][frozenset({player.name, GAME_MASTER_NAME})]  # type: ignore # noqa
    assert actual['chat_state'][frozenset({player.name, GAME_MASTER_NAME})].messages  # type: ignore # noqa
    assert actual['chat_state'][frozenset({player.name, GAME_MASTER_NAME})].messages[0].value.name == player.name  # type: ignore # noqa
    assert actual['chat_state'][frozenset({player.name, GAME_MASTER_NAME})].messages[0].value.message == expected  # type: ignore # noqa


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
def test_knight_act_in_night_integration() -> None:
    pass
