from functools import partial
from typing import Callable, Literal, Iterable
from langchain_core.runnables import Runnable
from langgraph.graph import Graph, StateGraph, START, END
from ..const import GAME_MASTER_NAME
from ..enums import EResult, ERole
from ..game_players.base import BaseGamePlayer
from ..models.state import (
    StateModel,
    create_dict_to_record_chat,
    create_dict_to_update_result,
    create_dict_without_state_updated,
)
from .utils import add_echo_node

# const
CHECK_VICTORY_CONDITION_TEARUP_NODE_NAME: str = 'check_victory_condition_tearup'  # noqa
CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME: str = 'check_victory_condition_teardown'  # noqa
CHECK_VICTORY_CONDITION_NODE_NAME: str = 'check_victory_condition'
RESULT_ANNOUNCE_NODE_NAME: str = 'announce_result'
REVEAL_ROLES_NODE_NAME: str = 'reveal_roles'

GAME_RESULT_MESSAGE_TEMPLATE: str = 'The game has ended: {result}'
PLAYER_ROLE_MESSAGE_TEMPLATE: str = '- The role of {name} is {role} (State: {state})'  # noqa
REVEAL_ALL_PLAYER_ROLES_MESSAGE_TEMPLATE: str = '''The roles of the players are as follows:
{roles}
'''  # noqa


def check_victory_condition(
    state: StateModel,
    players: Iterable[BaseGamePlayer],
) -> dict[str, EResult | None]:
    alive_players = [p for p in players if p.name in state.alive_players_names]  # noqa
    n_alive_players: int = len(state.alive_players_names)
    n_werewolves: int = len([
        player
        for player in alive_players
        if player.role == ERole.Werewolf
    ])
    n_villagers: int = n_alive_players - n_werewolves
    if n_werewolves == 0:
        return create_dict_to_update_result(EResult.VillagersWin)
    elif n_werewolves >= n_villagers:
        return create_dict_to_update_result(EResult.WerewolvesWin)
    else:
        return create_dict_to_update_result(None)


def create_check_victory_condition_subgraph(
    players: Iterable[BaseGamePlayer],
    *,
    echo_targets: list[Literal[  # type: ignore
        CHECK_VICTORY_CONDITION_TEARUP_NODE_NAME,  # type: ignore
        CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME,  # type: ignore
        CHECK_VICTORY_CONDITION_NODE_NAME,  # type: ignore
        RESULT_ANNOUNCE_NODE_NAME,  # type: ignore
        REVEAL_ROLES_NODE_NAME,  # type: ignore
    ] | str] = [
        RESULT_ANNOUNCE_NODE_NAME,
        REVEAL_ROLES_NODE_NAME,
    ],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    # define the graph
    workflow: Graph = StateGraph(StateModel)

    # define nodes
    workflow.add_node(
        CHECK_VICTORY_CONDITION_TEARUP_NODE_NAME,
        create_dict_without_state_updated,
    )
    workflow.add_node(
        CHECK_VICTORY_CONDITION_NODE_NAME,
        partial(check_victory_condition, players=players),
    )
    workflow.add_node(
        RESULT_ANNOUNCE_NODE_NAME,
        lambda state: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=GAME_RESULT_MESSAGE_TEMPLATE.format(result=state.result.value),  # noqa
        ),
    )
    workflow.add_node(
        REVEAL_ROLES_NODE_NAME,
        lambda state: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=REVEAL_ALL_PLAYER_ROLES_MESSAGE_TEMPLATE.format(
                roles='\n'.join([
                    PLAYER_ROLE_MESSAGE_TEMPLATE.format(
                        name=player.name,
                        role=player.role.value,
                        state=(
                            'Alive'
                            if player.name in state.alive_players_names else
                            'Excluded in ' + (
                                f'Day {list(v.value for v in state.daytime_vote_result_history).index(player.name)+1} daytime'  # noqa
                                if player.name in {v.value for v in state.daytime_vote_result_history} else  # noqa
                                f'Day {list(v.value for v in state.nighttime_vote_result_history).index(player.name)+1} nighttime'  # noqa
                            )
                        ),  # noqa
                    )
                    for player in players
                ])
            )
        ),
    )
    workflow.add_node(
        CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME,
        create_dict_without_state_updated,
    )
    # define edges
    workflow.add_edge(START, CHECK_VICTORY_CONDITION_TEARUP_NODE_NAME)
    workflow.add_edge(
        CHECK_VICTORY_CONDITION_TEARUP_NODE_NAME,
        CHECK_VICTORY_CONDITION_NODE_NAME,
    )
    workflow.add_conditional_edges(
        CHECK_VICTORY_CONDITION_NODE_NAME,
        lambda state: (
            CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME
            if state.result is None else
            RESULT_ANNOUNCE_NODE_NAME
        ),
        [
            RESULT_ANNOUNCE_NODE_NAME,
            CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME,
        ],
    )
    workflow.add_edge(
        RESULT_ANNOUNCE_NODE_NAME,
        REVEAL_ROLES_NODE_NAME,
    )
    workflow.add_edge(
        REVEAL_ROLES_NODE_NAME,
        CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME,
    )
    workflow.add_edge(
        CHECK_VICTORY_CONDITION_TEARDOWN_NODE_NAME,
        END,
    )
    # add display nodes
    add_echo_node(workflow, echo_targets, echo)
    return workflow
