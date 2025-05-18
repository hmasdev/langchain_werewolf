import os
from dotenv import load_dotenv
from flaky import flaky
from langchain_core.runnables import RunnableLambda
import pytest
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game_players.player_roles import Knight
from langchain_werewolf.models.state import StateModel, MsgModel

load_dotenv()


def test_knight_act_in_night() -> None:
    # preparation
    expected = 'Player0'
    player = Knight(
        name='Alice',
        runnable=RunnableLambda(lambda _: expected).with_types(output_type=str),  # noqa
    )
    players = [
        Knight(
            name=f'Player{i}',
            runnable=RunnableLambda(str),
        )
        for i in range(10)
    ]
    # execution
    actual = player.act_in_night(
        [player, *players],
        [],
        StateModel(alive_players_names=[player.name]+[p.name for p in players]),  # noqa
    )
    # assert
    assert actual == {'safe_players_names': {expected}}


@pytest.mark.integration
@pytest.mark.skipif(
    'OPENAI_API_KEY' not in os.environ,
    reason='OPENAI_API_KEY is not set.',
)
@flaky(max_runs=3, min_passes=1)
def test_knight_act_in_night_integration() -> None:
    # preparation
    from langchain_openai import ChatOpenAI
    from langchain_werewolf.game_players.helper import generate_game_player_runnable  # noqa
    player = Knight(
        name='Alice',
        runnable=generate_game_player_runnable(ChatOpenAI(model='gpt-4o-mini')),  # noqa
    )
    players = [
        Knight(
            name=f'Player{i}',
            runnable=RunnableLambda(str),
        )
        for i in range(10)
    ]
    # execution
    actual = player.act_in_night(
        [player, *players],
        [
            MsgModel(
                name=GAME_MASTER_NAME,
                message=f'Select one from {[p.name for p in players]}',  # noqa
            ),
        ],
        StateModel(alive_players_names=[player.name]+[p.name for p in players]),  # noqa
    )
    # assert
    assert actual['safe_players_names']
    assert list(actual['safe_players_names'])[0] in [p.name for p in players]  # type: ignore # noqa
