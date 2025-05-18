from functools import partial
from itertools import cycle
from typing import Callable, Generator, Iterable, Literal

from langchain_core.runnables import Runnable
from langgraph.graph import END, START, Graph, StateGraph
from pydantic import BaseModel, Field

from ..const import GAME_MASTER_NAME
from ..enums import ESpeakerSelectionMethod
from ..game_players import (
    BaseGamePlayerRole,
    find_player_by_name,
    is_werewolf_role,
)
from ..models.state import (
    ChatHistoryModel,
    StateModel,
    create_dict_to_record_chat,
    create_dict_to_update_chat_remaining_number,
    create_dict_to_update_current_speaker,
    get_related_messsages,
)
from ..utils import (
    random_permutated_infinite_generator,
)
from .prompts import (
    SYSTEM_PROMPT_TEMPLATE,
)
from .utils import create_message_history_prompt, add_echo_node  # noqa

# const
CHAT_TEARUP_NODE_NAME: str = 'tearup_chat'
CHAT_TEARDOWN_NODE_NAME: str = 'teardown_chat'
CHAT_SELECT_SPEAKER_NODE_NAME: str = 'select_speaker'
CHAT_NODE_NAME: str = 'chat'
UPDATE_N_CHAT_REMAINING_NODE_NAME: str = 'update_n_chat_remaining'

DAYTIME_DISCUSSION_PROMPT_TEMPLATE: str = '''=== Day {day}: Daytime ===
It is now noon. Werewolves are still in the village. Discussion begins to decide who to exclude from the game today.
--------------------
[Daytime Discussion]
Who should be excluded from the game in order to enable the villagers to win the game.
For example, the following behaviors are valid:
1. if you are in a villager team, pointing out who you think is a werewolf with the reasons
2. if you are in a werewolf team, pretending to be in a villager team
3. if you are in a werewolf team, NOT revealing your role
4. insisting on reasons why you should not be excluded from the game
5. lying about your role or hiding your role in order to enable your team to win the game.
Say logically and clearly.

Note that all roles are in this group chat and you are NOT the game master.
{alive_players_names} are alive in the game.
'''  # noqa

NIGHTTIME_DISCUSSION_PROMPT_TEMPLATE: str = '''=== Day {day}: Nighttime ===
It is now midnight. Discussion among werewolves begins to decide who to exclude from the game tonight.
--------------------
[Nighttime Discussion]
Who should be excluded from the game in order to enable the werewolves to win the game.
Note that there are only werewolves in this group chat.
However, {alive_players_names} are alive in the game.
'''  # noqa

ASK_TO_PLAYER_TO_SPEAK_PROMPT_TEMPLATE: str = '''
Generate a message which {name} will say. Output only the result.
You have enough time to think about what to say and say it.
Take a breath and think step by step to make a logical and clear statement.
'''  # noqa

speaker_selection_methods: dict[
    ESpeakerSelectionMethod,
    Callable[[Iterable[str]], Generator[str, None, None]],
] = {
    ESpeakerSelectionMethod.round_robin: cycle,  # type: ignore
    ESpeakerSelectionMethod.random: random_permutated_infinite_generator,
}


class GeneratePromptInputForChat(BaseModel):
    day: int = Field(..., title="the day number")
    alive_players_names: list[str] = Field(..., title="the names of the alive players")  # noqa


class GenerateSystemPromptInputForChat(BaseModel):
    name: str = Field(..., title="the name of the player")  # noqa
    messages: str = Field(..., title="the message history of the player")  # noqa


def _player_speak(
    state: StateModel,
    players: Iterable[BaseGamePlayerRole],
    participants: Iterable[str],
    generate_system_prompt: Callable[[GenerateSystemPromptInputForChat], str],
) -> dict[str, dict[frozenset[str], ChatHistoryModel]]:
    # validation
    if state.current_speaker is None:
        raise ValueError("current_speaker must not be None.")
    # initialize
    alive_players = [p for p in players if p.name in state.alive_players_names]  # noqa
    player = find_player_by_name(state.current_speaker, alive_players)  # noqa
    # get the related chat histories
    messages = get_related_messsages(player.name, state)
    # generate message
    message = player.generate_message(
        prompt=ASK_TO_PLAYER_TO_SPEAK_PROMPT_TEMPLATE.format(name=player.name),  # noqa
        system_prompt=generate_system_prompt(GenerateSystemPromptInputForChat(
            name=player.name,
            messages=create_message_history_prompt(messages),
        ))
    ).message
    # create a new chat history
    return create_dict_to_record_chat(
        player.name,
        list(participants)+[GAME_MASTER_NAME],
        message,
    )


def _tearup_chat(
    state: StateModel,
    players: Iterable[BaseGamePlayerRole],
    generate_prompt: Callable[[GeneratePromptInputForChat], str],
    n_turns_per_day: int,
) -> dict[str, object]:  # type: ignore
    return (  # type: ignore
        create_dict_to_update_chat_remaining_number(
            len([
                p for p in players
                if p.name in state.alive_players_names
            ]) * n_turns_per_day
        )
        | create_dict_to_record_chat(
            sender=GAME_MASTER_NAME,
            participants=[GAME_MASTER_NAME]+[p.name for p in players],
            message=generate_prompt(GeneratePromptInputForChat(
                day=state.day,
                alive_players_names=state.alive_players_names,
            )),
        )  # type: ignore
        | create_dict_to_update_current_speaker(None)  # type: ignore
    )


def create_run_chat_subbraph(
    players: Iterable[BaseGamePlayerRole],
    prompt: Callable[[GeneratePromptInputForChat], str] | str,
    system_prompt: Callable[[GenerateSystemPromptInputForChat], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    select_speaker: Callable[[Iterable[str]], Generator[str, None, None]] | type[cycle] | ESpeakerSelectionMethod = ESpeakerSelectionMethod.round_robin,  # noqa
    n_turns_per_day: int = 1,
    *,
    echo_targets: list[Literal[  # type: ignore
        CHAT_TEARUP_NODE_NAME,  # type: ignore
        CHAT_TEARDOWN_NODE_NAME,  # type: ignore
        CHAT_SELECT_SPEAKER_NODE_NAME,  # type: ignore
        CHAT_NODE_NAME,  # type: ignore
        UPDATE_N_CHAT_REMAINING_NODE_NAME,  # type: ignore
    ] | str] = [
        CHAT_SELECT_SPEAKER_NODE_NAME,
        CHAT_NODE_NAME,
    ],
    echo: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:

    if isinstance(select_speaker, ESpeakerSelectionMethod):
        select_speaker = speaker_selection_methods[select_speaker]
    speaker_generator = select_speaker([p.name for p in players])

    # define the graph
    workflow: Graph = StateGraph(StateModel)
    # define nodes
    workflow.add_node(
        CHAT_TEARUP_NODE_NAME,
        partial(
            _tearup_chat,
            players=players,
            generate_prompt=prompt if callable(prompt) else lambda m: prompt.format(**m.model_dump()),  # noqa
            n_turns_per_day=n_turns_per_day,
        ),
    )
    workflow.add_node(
        CHAT_SELECT_SPEAKER_NODE_NAME,
        lambda _: create_dict_to_update_current_speaker(
            speaker_generator.__next__(),
        ),
    )
    workflow.add_node(
        CHAT_NODE_NAME,
        partial(
            _player_speak,
            generate_system_prompt=(
                system_prompt
                if callable(system_prompt) else
                lambda m: system_prompt.format(**m.model_dump())
            ),
            participants=[player.name for player in players],
            players=players,
        )
    )
    workflow.add_node(
        UPDATE_N_CHAT_REMAINING_NODE_NAME,
        lambda state: create_dict_to_update_chat_remaining_number(state.n_chat_remaining-1),  # noqa
    )
    workflow.add_node(
        CHAT_TEARDOWN_NODE_NAME,
        lambda _: create_dict_to_update_current_speaker(None),
    )
    # define edges
    workflow.add_edge(START, CHAT_TEARUP_NODE_NAME)
    workflow.add_edge(CHAT_TEARUP_NODE_NAME, CHAT_SELECT_SPEAKER_NODE_NAME)
    workflow.add_conditional_edges(
        CHAT_SELECT_SPEAKER_NODE_NAME,
        lambda state: (
            CHAT_TEARDOWN_NODE_NAME
            if state.n_chat_remaining <= 0 else (
                CHAT_NODE_NAME
                if state.current_speaker in state.alive_players_names else
                CHAT_SELECT_SPEAKER_NODE_NAME
            )
        ),
        [CHAT_NODE_NAME, CHAT_SELECT_SPEAKER_NODE_NAME, CHAT_TEARDOWN_NODE_NAME],  # noqa
    )
    workflow.add_edge(CHAT_NODE_NAME, UPDATE_N_CHAT_REMAINING_NODE_NAME)
    workflow.add_edge(UPDATE_N_CHAT_REMAINING_NODE_NAME, CHAT_SELECT_SPEAKER_NODE_NAME)  # noqa
    workflow.add_edge(CHAT_TEARDOWN_NODE_NAME, END)

    # add display nodes
    add_echo_node(workflow, echo_targets, echo)

    return workflow


def create_run_daytime_chat_subgraph(
    players: Iterable[BaseGamePlayerRole],
    prompt: Callable[[GeneratePromptInputForChat], str] | str = DAYTIME_DISCUSSION_PROMPT_TEMPLATE,  # noqa
    system_prompt: Callable[[GenerateSystemPromptInputForChat], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    select_speaker: Callable[[Iterable[str]], Generator[str, None, None]] | ESpeakerSelectionMethod = ESpeakerSelectionMethod.round_robin,  # noqa
    n_turns_per_day: int = 1,
    *,
    display_targets: list[Literal[  # type: ignore
        CHAT_TEARUP_NODE_NAME,  # type: ignore
        CHAT_TEARDOWN_NODE_NAME,  # type: ignore
        CHAT_SELECT_SPEAKER_NODE_NAME,  # type: ignore
        CHAT_NODE_NAME,  # type: ignore
        UPDATE_N_CHAT_REMAINING_NODE_NAME,  # type: ignore
    ] | str] = [
        CHAT_SELECT_SPEAKER_NODE_NAME,
        CHAT_NODE_NAME,
    ],
    display: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    return create_run_chat_subbraph(
        players,
        prompt,
        system_prompt,
        select_speaker,
        n_turns_per_day=n_turns_per_day,
        echo_targets=display_targets,
        echo=display,
    )


def create_run_nighttime_chat_subgraph(
    werewolves: Iterable[BaseGamePlayerRole],
    prompt: Callable[[GeneratePromptInputForChat], str] | str = NIGHTTIME_DISCUSSION_PROMPT_TEMPLATE,  # noqa
    system_prompt: Callable[[GenerateSystemPromptInputForChat], str] | str = SYSTEM_PROMPT_TEMPLATE,  # noqa
    select_speaker: Callable[[Iterable[str]], Generator[str, None, None]] | ESpeakerSelectionMethod = ESpeakerSelectionMethod.round_robin,  # noqa
    n_turns_per_day: int = 1,
    *,
    display_targets: list[Literal[  # type: ignore
        CHAT_TEARUP_NODE_NAME,  # type: ignore
        CHAT_TEARDOWN_NODE_NAME,  # type: ignore
        CHAT_SELECT_SPEAKER_NODE_NAME,  # type: ignore
        CHAT_NODE_NAME,  # type: ignore
        UPDATE_N_CHAT_REMAINING_NODE_NAME,  # type: ignore
    ] | str] = [
        CHAT_SELECT_SPEAKER_NODE_NAME,
        CHAT_NODE_NAME,
    ],
    display: Callable[[StateModel], None] | Runnable[StateModel, None] | None = None,  # noqa
) -> Graph:
    # Check if `werewolves` contains only werewolf players
    invalid_players = [player.name for player in werewolves if not is_werewolf_role(player)]  # noqa
    if invalid_players:
        raise ValueError(f"The following players are not werewolves but participate in the nighttime chat: {', '.join(invalid_players)}.")  # noqa

    return create_run_chat_subbraph(
        werewolves,
        prompt,
        system_prompt,
        select_speaker,
        n_turns_per_day=n_turns_per_day,
        echo_targets=display_targets,
        echo=display,
    )
