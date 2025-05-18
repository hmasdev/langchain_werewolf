import os
from dotenv import load_dotenv
from flaky import flaky
from langchain_core.runnables import RunnableLambda
import pytest
from pytest_mock import MockerFixture
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game_players.player_roles import Knight
from langchain_werewolf.models.state import StateModel, MsgModel

load_dotenv()


def test_knight_act_in_night(mocker: MockerFixture) -> None:
    # mock
    mocker.patch("langchain_werewolf.game_players.player_roles.knight.extract_name", lambda msg, *args, **kwargs: msg)  # noqa
    # preparation
    expected_name = 'Player0'
    player = Knight(
        name='Alice',
        runnable=RunnableLambda(lambda _: expected_name).with_types(output_type=str),  # noqa
    )
    players = [
        Knight(
            name=f'Player{i}',
            runnable=RunnableLambda(str),
        )
        for i in range(10)
    ]
    expected = f'I decided to save {expected_name} in this night.'
    # execution
    actual = player.act_in_night(
        [player, *players],
        [],
        StateModel(alive_players_names=[player.name]+[p.name for p in players]),  # noqa
    )
    print(actual)
    # assert
    assert actual['safe_players_names'] == {expected_name}
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
@flaky(max_runs=5, min_passes=1)
def test_knight_act_in_night_integration() -> None:
    # preparation
    from langchain_openai import ChatOpenAI
    from langchain_werewolf.game_players import generate_game_player_runnable  # noqa
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
