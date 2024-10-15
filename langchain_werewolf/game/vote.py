from functools import partial
from logging import getLogger, Logger
from operator import attrgetter
from typing import Callable, Iterable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langgraph.graph import END, START, Graph, StateGraph
from pydantic import BaseModel, Field

from ..const import GAME_MASTER_NAME, DEFAULT_MODEL
from ..enums import ERole, ESide, ETimeSpan
from ..game_players.base import BaseGamePlayer
from ..llm_utils import extract_name
from ..models.state import (
    StateModel,
    create_dict_to_record_chat,
    create_dict_to_update_daytime_votes_current,
    create_dict_to_update_daytime_votes_history,
    create_dict_to_update_nighttime_votes_current,
    create_dict_to_update_nighttime_votes_history,
    create_dict_without_state_updated,
    get_related_messsages,
)
from .prompts import (
    SYSTEM_PROMPT_TEMPLATE,
)
from .utils import add_echo_node, create_message_history_prompt  # noqa

# const
VOTE_TEARUP_NODE_NAME: str = 'tearup_vote'
VOTE_TEARDOWN_NODE_NAME: str = 'teardown_vote'
VOTE_NODE_NAME_TEMPLATE: str = '{master}_ask_{name}_to_vote'

DAYTIME_VOTE_PROMPT_TEMPLATE: str = '''[Daytime Vote]
Who do you think should be excluded from the game?
Answer the question in order to help {player_side} to win the game.
Please select the name of the player from the following list:
{alive_players_names}
Think about who is in which role, reasoning step by step.
The conclusion must be simple and explicit like
- "PlayerX should be excluded."
'''

NIGHTTIME_VOTE_PROMPT_TEMPLATE: str = '''[Nighttime Vote]
Who do you think should be excluded from the game?
Answer the question in order to help {player_side} to win the game.
Please select the name of the player from the following list:
{alive_players_names}
Think about who is in which role, reasoning step by step.
The conclusion must be simple and explicit like
- "PlayerX should be excluded."
'''

NAME_EXTRACTION_CONTEXT_PROMPT: str = 'Extract the valid name of the player as the answer to "Who should be excluded from the game?"'  # noqa


class GeneratePromptInputForVote(BaseModel):
    player_role: ERole = Field(..., title="the role of the player")
    player_side: ESide = Field(..., title="the side of the player")
    alive_players_names: list[str] = Field(..., title="the names of the alive players")  # noqa


class GenerateSystemPromptInputForVote(BaseModel):
    name: str\
        = Field(..., title="the name of the player")
    messages: str\
        = Field(..., title="the message history of the player")


def _player_vote(
    state: StateModel,
    timespan: ETimeSpan,
    player: BaseGamePlayer,
    generate_system_prompt: Callable[[GenerateSystemPromptInputForVote], str],
    chat_model: BaseChatModel | str = DEFAULT_MODEL,
    seed: int | None = None,
    logger: Logger = getLogger(__name__),
) -> dict[str, object]:  # type: ignore

    # Case: When the player has been already excluded, he/she cannot vote
    if player.name not in state.alive_players_names:
        logger.info(f'{player.name} has been already excluded.')
        return create_dict_without_state_updated(state)

    update_votes_history: Callable[[dict[str, str]], dict[str, str]] = {  # type: ignore # noqa
        ETimeSpan.day: create_dict_to_update_daytime_votes_current,
        ETimeSpan.night: create_dict_to_update_nighttime_votes_current,
    }[timespan]

    # get the related chat histories
    messages = get_related_messsages(player.name, state)
    # generate message
    prompt = messages[-1].message if messages else ''
    message = player.generate_message(
        prompt=prompt,
        system_prompt=generate_system_prompt(GenerateSystemPromptInputForVote(
            name=player.name,
            messages=create_message_history_prompt(messages[:-1]),
        )),
    ).message
    name: str = extract_name(
        message,
        valid_names=state.alive_players_names,
        context=NAME_EXTRACTION_CONTEXT_PROMPT,
        chat_model=chat_model,
        seed=seed,
    )
    # create a new chat history
    votes_current: dict[str, str] = {
        ETimeSpan.day: state.daytime_votes_current,
        ETimeSpan.night: state.nighttime_votes_current,
    }[state.timespan]
    return (  # type: ignore
        create_dict_to_record_chat(player.name, [GAME_MASTER_NAME], message)
        | update_votes_history(votes_current | {player.name: name})  # type: ignore # noqa
    )


def _create_run_vote_subgraph(
    players: Iterable[BaseGamePlayer],
    timespan: ETimeSpan,
    prompt: Callable[[GeneratePromptInputForVote], str] | str,
    system_prompt: Callable[[GenerateSystemPromptInputForVote], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    chat_model: BaseChatModel | str = DEFAULT_MODEL,
    seed: int | None = None,
    *,
    echo_targets: list[Literal[  # type: ignore
        VOTE_TEARUP_NODE_NAME,  # type: ignore
        VOTE_TEARDOWN_NODE_NAME,  # type: ignore
    ]] = [],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
    logger: Logger = getLogger(__name__),
) -> Graph:
    # preprocess prompt
    prompt_func: Callable[[GeneratePromptInputForVote], str]
    if callable(prompt):
        prompt_func = prompt
    else:
        def prompt_func(m): return prompt.format(**m.model_dump())  # noqa
    system_prompt_func: Callable[[GenerateSystemPromptInputForVote], str]
    if callable(system_prompt):
        system_prompt_func = system_prompt
    else:
        def system_prompt_func(m): return system_prompt.format(**m.model_dump())  # noqa
    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # define nodes and edges
    workflow.add_node(
        VOTE_TEARUP_NODE_NAME,
        lambda state: (
            create_dict_to_update_daytime_votes_current({})
            | create_dict_to_update_nighttime_votes_current({})
            | create_dict_to_record_chat(
                sender=GAME_MASTER_NAME,
                participants=[GAME_MASTER_NAME]+[p.name for p in players],
                message=prompt_func(GeneratePromptInputForVote(
                    player_role=ERole.Villager,
                    player_side=ESide.Villager,
                    alive_players_names=[p for p in state.alive_players_names],  # noqa
                )),
            )
        ),
    )
    workflow.add_node(
        VOTE_TEARDOWN_NODE_NAME,
        RunnableBranch(
            (
                lambda state: state.timespan == ETimeSpan.day,
                RunnableLambda(attrgetter('daytime_votes_current'))
                | RunnableLambda(create_dict_to_update_daytime_votes_history),  # noqa
            ),
            (
                lambda state: state.timespan == ETimeSpan.night,
                RunnableLambda(attrgetter('nighttime_votes_current'))
                | RunnableLambda(create_dict_to_update_nighttime_votes_history),  # noqa
            ),
            lambda state: logger.error(f'Invalid timespan: {state.timespan}')
        ).with_types(input_type=StateModel, output_type=dict[str, object]),  # type: ignore # noqa
    )
    for player in players:
        workflow.add_node(
            VOTE_NODE_NAME_TEMPLATE.format(master=GAME_MASTER_NAME, name=player.name),  # noqa
            partial(
                _player_vote,
                timespan=timespan,
                player=player,
                generate_system_prompt=system_prompt_func,
                chat_model=chat_model,
                seed=seed,
            ),
        )
        workflow.add_edge(VOTE_TEARUP_NODE_NAME, VOTE_NODE_NAME_TEMPLATE.format(master=GAME_MASTER_NAME, name=player.name))  # noqa
        workflow.add_edge(VOTE_NODE_NAME_TEMPLATE.format(master=GAME_MASTER_NAME, name=player.name), VOTE_TEARDOWN_NODE_NAME)  # noqa
    workflow.add_edge(START, VOTE_TEARUP_NODE_NAME)
    workflow.add_edge(VOTE_TEARDOWN_NODE_NAME, END)
    # add display nodes
    add_echo_node(workflow, echo_targets, echo)
    return workflow


def create_vote_daytime_vote_subgraph(
    players: Iterable[BaseGamePlayer],
    prompt: Callable[[GeneratePromptInputForVote], str] | str = DAYTIME_VOTE_PROMPT_TEMPLATE,  # noqa
    system_prompt: Callable[[GenerateSystemPromptInputForVote], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    chat_model: BaseChatModel | str = DEFAULT_MODEL,
    seed: int | None = None,
    *,
    echo_targets: list[Literal[  # type: ignore
        VOTE_TEARUP_NODE_NAME,  # type: ignore
        VOTE_TEARDOWN_NODE_NAME,  # type: ignore
    ]] = [],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
    logger: Logger = getLogger(__name__),
) -> Graph:
    return _create_run_vote_subgraph(
        players,
        ETimeSpan.day,
        prompt,
        system_prompt,
        chat_model,
        seed,
        echo_targets=echo_targets,
        echo=echo,
        logger=logger,
    )


def create_vote_night_vote_subgraph(
    werewolves: Iterable[BaseGamePlayer],
    prompt: Callable[[GeneratePromptInputForVote], str] | str = NIGHTTIME_VOTE_PROMPT_TEMPLATE,  # noqa
    system_prompt: Callable[[GenerateSystemPromptInputForVote], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    chat_model: BaseChatModel | str = DEFAULT_MODEL,
    seed: int | None = None,
    *,
    echo_targets: list[Literal[  # type: ignore
        VOTE_TEARUP_NODE_NAME,  # type: ignore
        VOTE_TEARDOWN_NODE_NAME,  # type: ignore
    ]] = [],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
    logger: Logger = getLogger(__name__),
) -> Graph:
    return _create_run_vote_subgraph(
        werewolves,
        ETimeSpan.night,
        prompt,
        system_prompt,
        chat_model,
        seed,
        echo_targets=echo_targets,
        echo=echo,
        logger=logger,
    )
