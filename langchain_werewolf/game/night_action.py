from functools import partial
from typing import Iterable, Callable, Literal
from langchain_core.runnables import Runnable
from langgraph.graph import Graph, StateGraph, START, END
from pydantic import BaseModel, Field
from ..const import GAME_MASTER_NAME
from ..enums import ERole
from ..game_players.base import BaseGamePlayer
from ..game_players.helper import filter_state_according_to_player
from ..models.state import (
    ChatHistoryModel,
    StateModel,
    create_dict_to_record_chat,
    create_dict_without_state_updated,
    get_related_messsages,
)
from .utils import add_echo_node

# const
NIGHT_ACTION_TEARUP_NODE_NAME: str = 'tearup_night_action'
NIGHT_ACTION_TEARDOWN_NODE_NAME: str = 'teardown_night_action'
PASSTHROUGH_NODE_NAME_TEMPLATE: str = 'passthrough_{name}'
MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE: str = 'game_master_ask_{name}_to_act_in_night'  # noqa
ACTION_NODE_NAME_TEMPLATE: str = '{name}_night_action'

TEMPLATE_FOR_NIGHT_ACTION_TEMPLATE: str = '''
Your role is {role}.
Your avility is "{night_action}"
{question_to_decide_night_action}
Note that alive players are: {alive_players_names}
'''

NIGHTTIME_START_PROMPT_TEMPLATE: str = '''=== Day {day}: Nighttime Start ===
It is now midnight. Werewolves are active.
If you have an ability to act in the night, please act.
Otherwise, please wait in fear for the tomorrow.
Werewolves may exclude one player from the game.
'''


class GeneratePromptInputForNightAction(BaseModel):
    role: str = Field(..., title="the role of the player")
    night_action: str = Field(..., title="the night action of the player")  # noqa
    question_to_decide_night_action: str = Field(..., title="the question to decide the night action of the player")  # noqa
    alive_players_names: list[str] = Field(..., title="the names of the alive players")  # noqa


def _master_ask_player_to_act_in_night(
    state: StateModel,
    player: BaseGamePlayer,
    generate_prompt: Callable[[GeneratePromptInputForNightAction], str],
) -> dict[str, dict[frozenset[str], ChatHistoryModel]]:
    return create_dict_to_record_chat(
        sender=GAME_MASTER_NAME,
        participants=[player.name, GAME_MASTER_NAME],
        message=generate_prompt(
            GeneratePromptInputForNightAction(
                role=player.role.value,
                night_action=player.night_action or '',
                question_to_decide_night_action=player.question_to_decide_night_action or '',  # noqa
                alive_players_names=state.alive_players_names,
            ),
        ),
    )


def _player_act_in_night(
    state: StateModel,
    player: BaseGamePlayer,
    players: Iterable[BaseGamePlayer],
) -> dict[str, object]:
    return create_dict_without_state_updated(state) | player.act_in_night(
        players,
        get_related_messsages(player.name, state),
        filter_state_according_to_player(player, state),
    )


def _skip_player_act_in_night(
    state: StateModel,
    player: BaseGamePlayer,
    not_skip_destination_node_namd: str,
    skip_destination_node_name: str,
) -> str:
    if player.role == ERole.Werewolf:
        return skip_destination_node_name
    if player.name not in state.alive_players_names:
        return skip_destination_node_name
    return not_skip_destination_node_namd


def create_villagers_night_action_subgraph(
    players: Iterable[BaseGamePlayer],
    prompt: Callable[[GeneratePromptInputForNightAction], str] | str = TEMPLATE_FOR_NIGHT_ACTION_TEMPLATE,  # noqa
    *,
    echo_targets: list[Literal[  # type: ignore
        NIGHT_ACTION_TEARUP_NODE_NAME,  # type: ignore
        NIGHT_ACTION_TEARDOWN_NODE_NAME,  # type: ignore
        MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE,  # type: ignore
    ] | str] = [
        NIGHT_ACTION_TEARUP_NODE_NAME,
        MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE,
    ],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    # init
    if MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE in echo_targets:
        echo_targets = list(echo_targets)
        echo_targets.extend([
            MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE.format(name=player.name)  # noqa
            for player in players
        ])
        echo_targets.remove(MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE)
    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # define nodes and edges
    workflow.add_node(
        NIGHT_ACTION_TEARUP_NODE_NAME,
        lambda state: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=NIGHTTIME_START_PROMPT_TEMPLATE.format(day=state.day),
        ),
    )
    workflow.add_edge(
        START,
        NIGHT_ACTION_TEARUP_NODE_NAME,
    )
    for player in players:
        # define nodes
        workflow.add_node(
            PASSTHROUGH_NODE_NAME_TEMPLATE.format(name=player.name),
            create_dict_without_state_updated,
        )
        workflow.add_node(
            MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE.format(name=player.name),  # noqa
            partial(
                _master_ask_player_to_act_in_night,
                player=player,
                generate_prompt=prompt if callable(prompt) else lambda m: prompt.format(**m.model_dump()),  # noqa
            ),
        )
        workflow.add_node(
            ACTION_NODE_NAME_TEMPLATE.format(name=player.name),
            partial(_player_act_in_night, player=player, players=players),
        )
        # define edges
        workflow.add_edge(
            NIGHT_ACTION_TEARUP_NODE_NAME,
            PASSTHROUGH_NODE_NAME_TEMPLATE.format(name=player.name),
        )
        workflow.add_conditional_edges(
            PASSTHROUGH_NODE_NAME_TEMPLATE.format(name=player.name),
            partial(
                _skip_player_act_in_night,
                player=player,
                not_skip_destination_node_namd=MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE.format(name=player.name),  # noqa
                skip_destination_node_name=END,
            ),
            [MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE.format(name=player.name), END],  # noqa
        )
        workflow.add_edge(
            MASTER_ASK_PLAYER_TO_ACT_NODE_NAME_TEMPLATE.format(name=player.name),  # noqa
            ACTION_NODE_NAME_TEMPLATE.format(name=player.name),
        )
        workflow.add_edge(
            ACTION_NODE_NAME_TEMPLATE.format(name=player.name),
            END,
        )
    # add display nodes
    add_echo_node(workflow, echo_targets, echo)
    return workflow
