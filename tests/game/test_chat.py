from langchain_core.runnables import RunnableLambda
import pytest
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game.chat import _player_speak
from langchain_werewolf.game.prompts import SYSTEM_PROMPT_TEMPLATE
from langchain_werewolf.game_players import BaseGamePlayer, VILLAGER_ROLE
from langchain_werewolf.game_players.registry import PlayerRoleRegistry
from langchain_werewolf.models.state import StateModel


def test__player_speak() -> None:
    # preparation
    sender = 'player'
    message = 'message'
    state = StateModel(
        alive_players_names=[sender],
        current_speaker=sender,
    )
    player = PlayerRoleRegistry.create_player(
        name=sender,
        key=VILLAGER_ROLE,
        runnable=RunnableLambda(lambda _: message),
    )
    participants = frozenset([sender, 'another', GAME_MASTER_NAME])
    # execution
    actual = _player_speak(
        state,
        [player],
        participants,
        generate_system_prompt=lambda m: SYSTEM_PROMPT_TEMPLATE.format(**m.model_dump()),  # noqa
    )
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][participants]
    assert actual['chat_state'][participants].messages
    assert actual['chat_state'][participants].messages[0].value.name == sender  # noqa
    assert actual['chat_state'][participants].messages[0].value.message == message  # noqa
    assert actual['chat_state'][participants].messages[0].value.participants == participants  # noqa


@pytest.mark.parametrize(
    'current_speaker',
    [
        'current_speaker',
        None,
    ]
)
def test__player_speak_without_current_speaker(
    current_speaker: str | None,
) -> None:
    # preparation
    sender = 'player'
    message = 'message'
    state = StateModel(
        alive_players_names=[sender],
        current_speaker=current_speaker,
    )
    player = PlayerRoleRegistry.create_player(
        name=sender,
        key=VILLAGER_ROLE,
        runnable=RunnableLambda(lambda _: message),
    )
    participants = frozenset([sender, 'another', GAME_MASTER_NAME])
    # execution and assert
    with pytest.raises(ValueError):
        _player_speak(
            state,
            [player],
            participants,
            generate_system_prompt=lambda m: SYSTEM_PROMPT_TEMPLATE.format(**m.model_dump()),  # noqa
        )
