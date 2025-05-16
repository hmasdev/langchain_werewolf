from collections import Counter
from langchain_core.runnables import RunnableLambda
import pytest
from langchain_werewolf.const import GAME_MASTER_NAME
from langchain_werewolf.game_players import (
    BaseGamePlayer,
    PlayerRoleRegistry,
    is_player_with_role,
)
from langchain_werewolf.game_players.player_roles import (
    FortuneTeller,
    Knight,
    Villager,
    Werewolf,
)
from langchain_werewolf.game.setup import (
    _announce_game_rule,
    _announce_role,
    GAME_RULE_TEMPLATE,
    ROLE_ANNOUNCE_TEMPLATE,
    ROLE_EXPLANATION_TEMPLATE,
)
from langchain_werewolf.models.state import StateModel


def test__announce_game_rule() -> None:
    # preparation
    players = [
        PlayerRoleRegistry.create_player(name=f'player0', key=Villager.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player1', key=Werewolf.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player2', key=Knight.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player3', key=FortuneTeller.role, runnable=RunnableLambda(str)),  # noqa
    ]
    names = frozenset([GAME_MASTER_NAME]+[player.name for player in players])
    state = StateModel(alive_players_names=[player.name for player in players])  # noqa
    expected_message = GAME_RULE_TEMPLATE.format(
        n_roles=len({p.role for p in players}),
        roles='\n'.join([
            f'{idx+1}. {role_exp}'
            for idx, role_exp in enumerate({
                ROLE_EXPLANATION_TEMPLATE.format(
                    role=player.role,
                    side=player.side,
                    victory_condition=player.victory_condition,
                    night_action=player.night_action,
                )
                for player in players
            })
        ]),
        n_players_by_role_dict=dict(
            Counter([
                player.role
                for player in players
                if is_player_with_role(player)
            ])
        )
    )
    # execution
    actual = _announce_game_rule(state, players, GAME_RULE_TEMPLATE, ROLE_EXPLANATION_TEMPLATE)  # noqa
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][names]
    assert actual['chat_state'][names].messages
    assert actual['chat_state'][names].messages[0].value.name == GAME_MASTER_NAME  # noqa
    assert actual['chat_state'][names].messages[0].value.message == expected_message  # noqa
    assert actual['chat_state'][names].messages[0].value.participants == names  # noqa


@pytest.mark.parametrize(
    'player',
    [
        PlayerRoleRegistry.create_player(name=f'player0', key=Villager.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player1', key=Werewolf.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player2', key=Knight.role, runnable=RunnableLambda(str)),  # noqa
        PlayerRoleRegistry.create_player(name=f'player3', key=FortuneTeller.role, runnable=RunnableLambda(str)),  # noqa
    ]
)
def test__announce_role(player: BaseGamePlayer) -> None:
    # preparation
    state = StateModel(alive_players_names=[player.name])
    expected_message = ROLE_ANNOUNCE_TEMPLATE.format(
        role=player.role,
        side=player.side,
        victory_condition=player.victory_condition,
        night_action=player.night_action,
    )
    # execution
    actual = _announce_role(state, player, ROLE_ANNOUNCE_TEMPLATE)
    # assert
    assert actual['chat_state']
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])]
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages  # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.name == GAME_MASTER_NAME  # noqa
    assert actual['chat_state'][frozenset([player.name, GAME_MASTER_NAME])].messages[0].value.message == expected_message  # noqa
