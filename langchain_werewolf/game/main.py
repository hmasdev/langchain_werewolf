from operator import attrgetter
from typing import Iterable, Callable
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from ..enums import ERole, ETimeSpan
from ..game_players.base import BaseGamePlayer
from ..models.state import (
    StateModel,
    create_dict_to_update_day,
    create_dict_to_update_timespan,
    create_dict_to_reset_state,
    create_dict_without_state_updated,
)
from .chat import create_run_daytime_chat_subgraph, create_run_nighttime_chat_subgraph  # noqa
from .check_result import create_check_victory_condition_subgraph
from .elimination import create_elimination_subgraph
from .night_action import create_villagers_night_action_subgraph
from .setup import create_game_preparation_graph
from .vote import create_vote_daytime_vote_subgraph, create_vote_night_vote_subgraph  # noqa


def create_game_graph(
    players: Iterable[BaseGamePlayer],
    preparation_kwargs: dict[str, object] = {},
    check_victory_condition_kwargs: dict[str, object] = {},
    check_victory_condition_before_daytime_kwargs: dict[str, object] = {},
    check_victory_condition_before_nighttime_kwargs: dict[str, object] = {},
    chat_kwargs: dict[str, object] = {},
    daytime_chat_kwargs: dict[str, object] = {},
    nighttime_chat_kwargs: dict[str, object] = {},
    vote_kwargs: dict[str, object] = {},
    daytime_vote_kwargs: dict[str, object] = {},
    nighttime_vote_kwargs: dict[str, object] = {},
    night_action_kwargs: dict[str, object] = {},
    elimination_kwargs: dict[str, object] = {},
    elimination_after_daytime_vote_kwargs: dict[str, object] = {},
    elimination_after_night_vote_kwargs: dict[str, object] = {},
    echo: Runnable[StateModel, None] | Callable[[StateModel], None] | None = None,  # noqa
) -> CompiledGraph:
    # preparation
    players = list(players)
    werewolves = [player for player in players if player.role == ERole.Werewolf]  # noqa
    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # add nodes
    workflow.add_node(
        'game_preparation',
        create_game_preparation_graph(
            players,
            **preparation_kwargs,  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'add_day',
        RunnableLambda(attrgetter('day'))
        | RunnableLambda(lambda x: x + 1)
        | RunnableLambda(create_dict_to_update_day)
    )
    workflow.add_node(
        'setup_daytime',
        RunnableParallel(
            timespan=RunnableLambda(lambda _: create_dict_to_update_timespan(ETimeSpan.day)),  # noqa
            reset=RunnableLambda(create_dict_to_reset_state),
        )
        | RunnableLambda(lambda dic: dic['timespan'] | dic['reset'])
    )
    workflow.add_node(
        'setup_nighttime',
        RunnableParallel(
            timespan=RunnableLambda(lambda _: create_dict_to_update_timespan(ETimeSpan.night)),  # noqa
            reset=RunnableLambda(create_dict_to_reset_state),
        )
        | RunnableLambda(lambda dic: dic['timespan'] | dic['reset'])
    )
    workflow.add_node(
        'check_victory_condition_before_nighttime',
        create_check_victory_condition_subgraph(
            players,
            **(check_victory_condition_kwargs | check_victory_condition_before_nighttime_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'check_victory_condition_before_daytime',
        create_check_victory_condition_subgraph(
            players,
            **(check_victory_condition_kwargs | check_victory_condition_before_daytime_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'daytime_chat',
        create_run_daytime_chat_subgraph(
            players,
            **(chat_kwargs | daytime_chat_kwargs),  # type: ignore # noqa
            display=echo,
        ).compile(),
    )
    workflow.add_node(
        'night_chat',
        create_run_nighttime_chat_subgraph(
            werewolves,
            **(chat_kwargs | nighttime_chat_kwargs),  # type: ignore # noqa
            display=echo,
        ).compile(),
    )
    workflow.add_node(
        'daytime_vote',
        create_vote_daytime_vote_subgraph(
            players,
            **(vote_kwargs | daytime_vote_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'night_vote',
        create_vote_night_vote_subgraph(
            werewolves,
            **(vote_kwargs | nighttime_vote_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'villagers_night_action',
        create_villagers_night_action_subgraph(
            players,
            **night_action_kwargs,  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'elimination_after_daytime_vote',
        create_elimination_subgraph(
            players,
            **(elimination_kwargs | elimination_after_daytime_vote_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'elimination_after_night_vote',
        create_elimination_subgraph(
            players,
            **(elimination_kwargs | elimination_after_night_vote_kwargs),  # type: ignore # noqa
            echo=echo,
        ).compile(),
    )
    workflow.add_node(
        'state_validation_before_daytime',
        RunnableLambda(lambda state: StateModel.validate_state(state) and create_dict_without_state_updated(state)),  # noqa
    )
    workflow.add_node(
        'state_validation_before_nighttime',
        RunnableLambda(lambda state: StateModel.validate_state(state) and create_dict_without_state_updated(state)),  # noqa
    )

    workflow.add_edge(START, 'game_preparation')
    workflow.add_edge('game_preparation', 'check_victory_condition_before_daytime')  # noqa
    workflow.add_conditional_edges(
        'check_victory_condition_before_daytime',
        lambda state: 'add_day' if state.result is None else END,
        ['add_day', END],
    )
    workflow.add_conditional_edges(
        # emergency exit
        'add_day',
        lambda state: 'setup_daytime' if state.day < len(players) else END,  # noqa
        ['setup_daytime', END],
    )
    workflow.add_edge('setup_daytime', 'state_validation_before_daytime')
    workflow.add_edge('state_validation_before_daytime', 'daytime_chat')
    workflow.add_edge('daytime_chat', 'daytime_vote')
    workflow.add_edge('daytime_vote', 'elimination_after_daytime_vote')
    workflow.add_edge('elimination_after_daytime_vote', 'check_victory_condition_before_nighttime')  # noqa
    workflow.add_conditional_edges(
        'check_victory_condition_before_nighttime',
        lambda state: 'setup_nighttime' if state.result is None else END,  # noqa
        ['setup_nighttime', END],
    )
    workflow.add_edge('setup_nighttime', 'state_validation_before_nighttime')
    workflow.add_edge('state_validation_before_nighttime', 'villagers_night_action')  # noqa
    workflow.add_edge('villagers_night_action', 'night_chat')
    workflow.add_edge('night_chat', 'night_vote')
    workflow.add_edge('night_vote', 'elimination_after_night_vote')
    workflow.add_edge('elimination_after_night_vote', 'check_victory_condition_before_daytime')  # noqa

    return workflow.compile()
