from collections import Counter
from typing import Iterable, Callable, Literal
from langchain_core.runnables import Runnable
from langgraph.graph import Graph, StateGraph, START, END
from ..const import GAME_MASTER_NAME
from ..enums import ETimeSpan
from ..game_players.base import BaseGamePlayer
from ..models.state import (
    StateModel,
    create_dict_to_record_chat,
    create_dict_to_update_alive_players,
    create_dict_to_update_daytime_vote_result_history,
    create_dict_to_update_nighttime_vote_result_history,
    create_dict_without_state_updated,
)
from .utils import add_echo_node

# const
ELIMINATE_TEARUP_NODE_NAME: str = 'elimination_tearup'
ELIMINATE_TEARDOWN_NODE_NAME: str = 'elimination_teardown'
ELIMINATE_NODE_NAME: str = 'elmination'
ANNOUNCE_NODE_NAME_DAYTIME: str = 'elimination_announce_daytime'
ANNOUNCE_NODE_NAME_NIGHTTIME: str = 'elimination_announce_nighttime'
REPLY_NODE_NAME_TEMPLATE: str = 'reply_from_{name}_for_elimination'

ANNOUNCE_MESSAGE_TEMPLATE: str = '''
{last_vote_result} has been eliminated.
Last votes are follows: {last_votes}.
'''


def _eliminate_player(
    state: StateModel,
    select_from_same_votes: Callable[[Iterable[str]], str | None] = lambda x: list(x)[0],  # noqa
    # FIXME: list(x)[0] is not a good default value
) -> dict:
    # perparation
    # TODO: error handling for timespan
    create_dict_to_update_vote_result_history: Callable[[str | None], dict[str, object]]  # noqa
    if state.timespan == ETimeSpan.day:
        latest_votes = state.daytime_votes_history[-1].value
        create_dict_to_update_vote_result_history = create_dict_to_update_daytime_vote_result_history  # type: ignore # noqa
    else:
        latest_votes = state.nighttime_votes_history[-1].value
        create_dict_to_update_vote_result_history = create_dict_to_update_nighttime_vote_result_history  # type: ignore # noqa
    # filter out the invalid votes
    valid_votes = {
        voter: voted
        for voter, voted in latest_votes.items()
        if all([
            voter in state.alive_players_names,
            voted in state.alive_players_names,
            voted not in state.safe_players_names,
        ])
    }
    if len(valid_votes) == 0:
        return create_dict_to_update_vote_result_history(None)
    # count the votes
    votes_count = Counter(valid_votes.values())
    max_n_votes = max(votes_count.values())
    # get the candidates with the max votes
    candidates = [
        player
        for player, n_votes in votes_count.items()
        if n_votes == max_n_votes
    ]
    # select the player to eliminate
    eliminated = select_from_same_votes(candidates)
    return (
        create_dict_to_update_vote_result_history(eliminated)
        | create_dict_to_update_alive_players(
            [n for n in state.alive_players_names if n != eliminated]  # noqa
        )
    )


def create_elimination_subgraph(
    players: Iterable[BaseGamePlayer],
    *,
    echo_targets: list[Literal[  # type: ignore
        ANNOUNCE_NODE_NAME_DAYTIME,  # type: ignore
        ANNOUNCE_NODE_NAME_NIGHTTIME,  # type: ignore
        ELIMINATE_NODE_NAME,  # type: ignore
    ] | str] = [
        ANNOUNCE_NODE_NAME_DAYTIME,
        ANNOUNCE_NODE_NAME_NIGHTTIME,
        ELIMINATE_NODE_NAME,
    ],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # define nodes
    workflow.add_node(ELIMINATE_TEARUP_NODE_NAME, create_dict_without_state_updated)  # noqa
    workflow.add_node(ELIMINATE_TEARDOWN_NODE_NAME, create_dict_without_state_updated)  # noqa
    workflow.add_node(ELIMINATE_NODE_NAME, _eliminate_player)
    workflow.add_node(
        ANNOUNCE_NODE_NAME_DAYTIME,
        lambda state: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=ANNOUNCE_MESSAGE_TEMPLATE.format(
                last_vote_result=state.daytime_vote_result_history[-1].value or 'No one',  # noqa
                last_votes=state.daytime_votes_history[-1].value,
            ),
            # TODO: error handling for no vote history case
        ),
    )
    workflow.add_node(
        ANNOUNCE_NODE_NAME_NIGHTTIME,
        lambda state: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=ANNOUNCE_MESSAGE_TEMPLATE.format(
                last_vote_result=state.nighttime_vote_result_history[-1].value or 'No one',  # noqa
                last_votes='-',
            ),
            # TODO: error handling for no vote history case
        ),
    )
    # add edges
    workflow.add_edge(START, ELIMINATE_TEARUP_NODE_NAME)
    workflow.add_edge(
        ELIMINATE_TEARUP_NODE_NAME,
        ELIMINATE_NODE_NAME,
    )
    workflow.add_conditional_edges(
        ELIMINATE_NODE_NAME,
        lambda state: ANNOUNCE_NODE_NAME_DAYTIME if state.timespan == ETimeSpan.day else ANNOUNCE_NODE_NAME_NIGHTTIME,  # noqa
        [ANNOUNCE_NODE_NAME_DAYTIME, ANNOUNCE_NODE_NAME_NIGHTTIME],
    )
    workflow.add_edge(
        ANNOUNCE_NODE_NAME_DAYTIME,
        ELIMINATE_TEARDOWN_NODE_NAME,
    )
    workflow.add_edge(
        ANNOUNCE_NODE_NAME_NIGHTTIME,
        ELIMINATE_TEARDOWN_NODE_NAME,
    )
    workflow.add_edge(ELIMINATE_TEARDOWN_NODE_NAME, END)
    # add display nodes
    add_echo_node(workflow, echo_targets, echo)

    return workflow
