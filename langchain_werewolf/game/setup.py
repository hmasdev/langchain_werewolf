from collections import Counter
from functools import partial
from typing import Callable, Iterable, Literal
from langchain_core.runnables import Runnable
from langgraph.graph import Graph, StateGraph, START, END
from ..const import GAME_MASTER_NAME, PACKAGE_NAME
from ..game_players import (
    BaseGamePlayer,
    is_player_with_role,
    is_player_with_side,
)
from ..models.state import (
    ChatHistoryModel,
    StateModel,
    create_dict_to_record_chat,
)
from .utils import add_echo_node

# const
WELCOME_NODE_NAME: str = "welcome_to_game"
RULE_ANNOUNCE_NODE_NAME: str = "rule_announcement"  # noqa
ROLE_ANNOUNCE_NODE_NAME_TEMPLATE: str = "role_announcement_{name}"  # noqa


WELCOME_TO_GAME_MESSAGE: str = f' WELCOME TO {PACKAGE_NAME} GAME '

# GAME_RULE_TEMPLATE is used to explain the game rule.
GAME_RULE_TEMPLATE: str = '''====== Game Rule ======
This is a werewolf game.
There are two teams: villagers and werewolves.
The villagers win if all werewolves are exclude from the game.
The werewolves win if they equal or outnumber half of the total number of players.,
There are {n_roles} roles in this game.

{roles}

The game repeats the following steps until the villagers win or the werewolves win.,
1. Daytime discussion
2. Vote to exclude from the game
3. Check victory condition
4. Nighttime action
5. Check victory condition

NOTE:
- If there is more than one player who received the most votes in the voting for the person to be excluded, one of them will be chosen at random to be excluded.,
- You can lie or hide your role in the daytime discussion.
- The number of each role is as follows (Format: {{role}}: {{number of players}}): {n_players_by_role_dict}
=======================
'''  # noqa

ROLE_EXPLANATION_TEMPLATE: str = '''Role: {role}
   - Side: {side}
   - Victory condition: {victory_condition}
   - Night action: {night_action}
'''

ROLE_ANNOUNCE_TEMPLATE: str = '''====== Role Announcement ======
You are a player with a role "{role}" in a werewolf game.
Your victory condition is: "{victory_condition}".
Your night action is: "{night_action}"
How will you behave in order to enable your team({side}) to win the game?
'''


def _announce_game_rule(
    state: StateModel,
    players: Iterable[BaseGamePlayer],
    game_rule_template: str,
    role_explanation_template: str,
) -> dict[str, dict[frozenset[str], ChatHistoryModel]]:

    roles_explanation = {
        role_explanation_template.format(
            role=is_player_with_role(player) and player.role,  # noqa
            side=is_player_with_side(player) and player.side,  # noqa
            victory_condition=is_player_with_side(player) and player.victory_condition,  # noqa
            night_action=is_player_with_role(player) and player.night_action,
            # NOTE: is_player_with_xxx is used for type guard # FIXME
        )
        for player in players
    }

    return create_dict_to_record_chat(
        sender=GAME_MASTER_NAME,
        participants=[GAME_MASTER_NAME]+[player.name for player in players],
        message=game_rule_template.format(
            n_roles=len(roles_explanation),
            roles='\n'.join([
                f'{idx+1}. {role_exp}'
                for idx, role_exp in enumerate(roles_explanation)
            ]),
            n_players_by_role_dict=dict(
                Counter([
                    player.role
                    for player in players
                    if is_player_with_role(player)
                ])
            )
        ),
    )


def _announce_role(
    state: StateModel,
    player: BaseGamePlayer,
    role_announce_template: str,
) -> dict[str, dict[frozenset[str], ChatHistoryModel]]:
    return create_dict_to_record_chat(
        sender=GAME_MASTER_NAME,
        participants=[player.name, GAME_MASTER_NAME],
        message=role_announce_template.format(
            role=is_player_with_role(player) and player.role,
            side=is_player_with_side(player) and player.side,
            victory_condition=is_player_with_side(player) and player.victory_condition,  # noqa
            night_action=is_player_with_role(player) and player.night_action,  # noqa
            # NOTE: is_player_with_xxx is used for type guard # FIXME
        ),
    )


def create_game_preparation_graph(
    players: Iterable[BaseGamePlayer],
    game_rule_template: str = GAME_RULE_TEMPLATE,
    role_explanation_template: str = ROLE_EXPLANATION_TEMPLATE,
    role_announce_template: str = ROLE_ANNOUNCE_TEMPLATE,
    *,
    echo_targets: Iterable[Literal[  # type: ignore
        ROLE_ANNOUNCE_NODE_NAME_TEMPLATE,  # type: ignore
        RULE_ANNOUNCE_NODE_NAME,  # type: ignore
        WELCOME_NODE_NAME,  # type: ignore
    ] | str] = [
        RULE_ANNOUNCE_NODE_NAME,
        WELCOME_NODE_NAME,
    ],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    # init
    # replace template node name with specific node names
    if ROLE_ANNOUNCE_NODE_NAME_TEMPLATE in echo_targets:
        echo_targets = list(echo_targets)
        echo_targets.extend([
            ROLE_ANNOUNCE_NODE_NAME_TEMPLATE.format(name=player.name)
            for player in players
        ])
        echo_targets.remove(ROLE_ANNOUNCE_NODE_NAME_TEMPLATE)
    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # add nodes and edges
    workflow.add_node(
        WELCOME_NODE_NAME,
        lambda _: create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message='\n'.join([
                (len(WELCOME_TO_GAME_MESSAGE) + 2) * '=',
                '=' + WELCOME_TO_GAME_MESSAGE + '=',
                (len(WELCOME_TO_GAME_MESSAGE) + 2) * '=',
            ]),
        ),
    )
    workflow.add_node(
        RULE_ANNOUNCE_NODE_NAME,
        partial(
            _announce_game_rule,
            players=players,
            game_rule_template=game_rule_template,
            role_explanation_template=role_explanation_template
        ),
    )
    workflow.add_edge(START, WELCOME_NODE_NAME)
    workflow.add_edge(WELCOME_NODE_NAME, RULE_ANNOUNCE_NODE_NAME)
    for player in players:
        # Role Announcement
        workflow.add_node(
            ROLE_ANNOUNCE_NODE_NAME_TEMPLATE.format(name=player.name),  # noqa
            partial(
                _announce_role,
                player=player,
                role_announce_template=role_announce_template,
            ),
        )
        # Edge
        workflow.add_edge(
            RULE_ANNOUNCE_NODE_NAME,
            ROLE_ANNOUNCE_NODE_NAME_TEMPLATE.format(name=player.name),
        )
        workflow.add_edge(
            ROLE_ANNOUNCE_NODE_NAME_TEMPLATE.format(name=player.name),
            END,
        )

    # add display nodes
    add_echo_node(workflow, echo_targets, echo)

    return workflow
