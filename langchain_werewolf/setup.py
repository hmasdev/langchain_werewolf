from collections import Counter
from functools import partial
from itertools import chain, cycle
from logging import getLogger, Logger
from operator import attrgetter
import random
from typing import Callable, Iterable
import click
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from .const import (
    BASE_LANGUAGE,
    CLI_PROMPT_COLOR,
    CLI_PROMPT_SUFFIX,
    CLI_ECHO_COLORS,
    DEFAULT_MODEL,
    DEFAULT_PLAYER_PREFIX,
    GAME_MASTER_NAME,
    MODEL_SERVICE_MAP,
)
from .enums import (
    ERole,
    ESystemOutputType,
    EChatService,
    ELanguage,
    EInputOutputType,
)
from .game_players.base import BaseGamePlayer
from .game_players.helper import generate_game_player_runnable, filter_state_according_to_player  # noqa
from .io import create_input_runnable, create_output_runnable
from .llm_utils import create_chat_model, create_translator_runnable
from .models.config import PlayerConfig
from .models.state import (
    IdentifiedModel,
    MsgModel,
    StateModel,
    get_related_messsages_with_id,
)
from .utils import consecutive_string_generator


def _generate_base_runnable(
    model: str | None,
    input_output_type: EInputOutputType | None,
    seed: int | None = None,
) -> BaseChatModel | Runnable[str, str]:
    if model is None:
        return create_chat_model(
            DEFAULT_MODEL,
            seed=seed if seed is not None and seed >= 0 else None,
        )
    elif (
        model in MODEL_SERVICE_MAP
        and MODEL_SERVICE_MAP[model] in {
            EChatService.OpenAI,
            EChatService.Google,
            EChatService.Groq,
        }
    ):
        return create_chat_model(
            model,
            seed=seed if seed is not None and seed >= 0 else None,
        )
    elif (
        model in MODEL_SERVICE_MAP
        and MODEL_SERVICE_MAP[model] == EChatService.CLI
        and input_output_type is not None
    ):
        return create_input_runnable(
            input_func=input_output_type,
            styler=partial(click.style, fg=CLI_PROMPT_COLOR),
            prompt_suffix=CLI_PROMPT_SUFFIX,
        )
    else:
        raise ValueError(f'Unsupported: model={model}, input_output_type={input_output_type}')  # noqa


def generate_players(
    n_players: int,
    n_werewolves: int,
    n_knights: int,
    n_fortune_tellers: int,
    custom_players: list[PlayerConfig] = [],
    model: str | None = None,
    seed: int = -1,
    input_output_type: EInputOutputType | None = None,
    logger: Logger = getLogger(__name__),
) -> list[BaseGamePlayer]:

    logger.info(f'n_players: {n_players}')
    logger.info(f'n_werewolves: {n_werewolves}')
    logger.info(f'n_knights: {n_knights}')
    logger.info(f'n_fortune_tellers: {n_fortune_tellers}')
    logger.info(f'seed: {seed}')
    logger.info(f'len(custom_players): {len(custom_players)}')

    # initialize
    if n_players < n_werewolves:
        raise ValueError(f'The number of players ({n_players}) is less than the number of werewolves ({n_werewolves}).')  # noqa
    if n_players < n_knights:
        raise ValueError(f'The number of players ({n_players}) is less than the number of knights ({n_knights}).')  # noqa
    if n_players < n_fortune_tellers:
        raise ValueError(f'The number of players ({n_players}) is less than the number of fortune tellers ({n_fortune_tellers}).')  # noqa
    n_villagers = n_players - n_werewolves - n_knights - n_fortune_tellers
    if n_villagers < 0:
        raise ValueError(f'n_players - n_werewolves - n_knights - n_fortune_tellers = {n_players} - {n_werewolves} - {n_knights} - {n_fortune_tellers} is less than 0.')  # noqa

    # validate the number of players
    if n_players < len(custom_players):
        raise ValueError(f'The number of players ({n_players}) is less than the number of custom players ({len(custom_players)}).')  # noqa

    # create config of players
    players_cfg = custom_players[::] + [None] * (n_players-len(custom_players))  # noqa
    roles = [player.role if player else None for player in players_cfg]  # noqa

    # validate the number of roles
    counter = Counter(roles)
    if n_werewolves < counter.get(ERole.Werewolf, 0):
        raise ValueError(f'The number of werewolves ({n_werewolves}) is less than the number of custom werewolves ({counter.get(ERole.Werewolf, 0)}).')  # noqa
    if n_knights < counter.get(ERole.Knight, 0):
        raise ValueError(f'The number of knights ({n_knights}) is less than the number of custom knights ({counter.get(ERole.Knight, 0)}).')  # noqa
    if n_fortune_tellers < counter.get(ERole.FortuneTeller, 0):
        raise ValueError(f'The number of fortune tellers ({n_fortune_tellers}) is less than the number of custom fortune tellers ({counter.get(ERole.FortuneTeller, 0)}).')  # noqa
    if n_villagers < counter.get(ERole.Villager, 0):
        raise ValueError(f'The number of villagers ({n_villagers}) is less than the number of custom villagers ({counter.get(ERole.Villager, 0)}).')  # noqa

    # randome roles
    generated_roles: list[ERole] = list(chain(
        [ERole.Werewolf] * (n_werewolves-counter.get(ERole.Werewolf, 0)),
        [ERole.Knight] * (n_knights-counter.get(ERole.Knight, 0)),
        [ERole.FortuneTeller] * (n_fortune_tellers-counter.get(ERole.FortuneTeller, 0)),  # noqa
        [ERole.Villager] * (n_villagers-counter.get(ERole.Villager, 0)),
    ))
    random.shuffle(generated_roles)

    # generate players
    name_generator = consecutive_string_generator(DEFAULT_PLAYER_PREFIX)
    players = [
        BaseGamePlayer.instantiate(
            role=player_cfg.role if player_cfg and player_cfg.role else generated_roles.pop(),  # noqa
            name=player_cfg.name if player_cfg and player_cfg.name else name_generator.__next__(),  # noqa
            runnable=generate_game_player_runnable(_generate_base_runnable(
                player_cfg.model if hasattr(player_cfg, 'model') else model,  # type: ignore # noqa
                player_cfg.player_input_interface if hasattr(player_cfg, 'input_output_type') else input_output_type,  # type: ignore # noqa
                seed
            )),
            output=(
                create_output_runnable(player_cfg.player_output_interface)
                if player_cfg and player_cfg.player_output_interface
                else None
            ),
            formatter=player_cfg.formatter if player_cfg and player_cfg.formatter else None,  # noqa
        )
        for player_cfg in players_cfg
    ]
    # Internal Error
    assert len(players) == n_players
    assert all(player.ready() for player in players), players
    return players


def _create_echo_runnable_by_player(
    player: BaseGamePlayer,
    *,
    player_config: PlayerConfig | None = None,
    model: str = DEFAULT_MODEL,
    input_output_type: EInputOutputType | None = None,
    cache: set[str] | None = None,
    color: str | None = None,
    language: ELanguage = BASE_LANGUAGE,
    seed: int = -1,
) -> Runnable[StateModel, None]:
    cache = cache or set()  # NOTE: if cache is None, cache does not work
    # create runnable
    return (
        RunnableLambda(lambda state: filter_state_according_to_player(player, state))  # noqa
        | RunnableLambda(lambda state: get_related_messsages_with_id(player.name, state))  # noqa
        | RunnableBranch(
            (
                lambda msg: msg.id not in cache,
                RunnableParallel(
                    to_cache=RunnableLambda(attrgetter('id')) | RunnableLambda(cache.add),  # noqa
                    echo=(
                        RunnableLambda(attrgetter('value'))
                        | RunnableParallel(
                            orig=RunnablePassthrough(),
                            translated_msg=(
                                RunnableLambda(attrgetter('message'))
                                | create_translator_runnable(
                                    to_language=language,
                                    chat_llm=_generate_base_runnable(
                                        player_config.model if hasattr(player_config, 'model') else model,  # type: ignore # noqa
                                        player_config.player_output_interface if hasattr(player_config, 'input_output_type') else input_output_type,  # type: ignore # noqa
                                        seed=seed
                                    ),
                                )
                            ),
                        )
                        | RunnableLambda(lambda dic: MsgModel(**(dic['orig'].model_dump() | {'message': dic['translated_msg']})))  # noqa
                        | create_output_runnable(
                            output_func=player.receive_message,
                            styler=partial(click.style, fg=color or CLI_PROMPT_COLOR),  # noqa
                        )
                    ),
                )
            ),
            RunnablePassthrough(),
        ).with_types(input_type=IdentifiedModel[MsgModel]).with_config({'max_concurrency': 1}).map()  # noqa
        | RunnableLambda(lambda _: None)
    )


def _create_echo_runnable_by_system(
    kind: EInputOutputType,
    level: ESystemOutputType | str,
    *,
    model: str = DEFAULT_MODEL,
    player_names: list[str] | None = None,
    cache: set[str] | None = None,
    color: str | dict[str, str] | None = None,
    language: ELanguage = BASE_LANGUAGE,
    formatter: Callable[[MsgModel], str] | str | None = None,
    seed: int = -1,
) -> Runnable[StateModel, None]:
    # initialize
    player_names = player_names or []
    cache = cache or set()  # NOTE: if cache is None, cache does not work
    try:
        _system_related_dict: dict[ESystemOutputType | str, str | set[str] | None] = {  # noqa
            ESystemOutputType.off: None,
            ESystemOutputType.all: GAME_MASTER_NAME,
            ESystemOutputType.public: set([GAME_MASTER_NAME]+player_names),
        } | {name: name for name in player_names}
        system_related: str | set[str] | None = _system_related_dict[level]  # noqa
    except KeyError:
        raise ValueError(f'Invalid level: {level}. Valid levels are {list(_system_related_dict.keys())}')  # noqa
    # create runnable
    if system_related is None:
        return RunnableLambda(lambda _: None)
    # preprocess formatter
    formatter = formatter or MsgModel.format
    formatter_runnable: Runnable[MsgModel, str]
    if isinstance(formatter, str):
        formatter_runnable = (
            RunnableLambda(MsgModel.model_dump)
            | RunnableLambda(lambda dic: formatter.format(**dic))
        )
    else:
        formatter_runnable = RunnableLambda(formatter)
    formatter_runnable = (
        RunnableParallel(
            orig=RunnablePassthrough(),
            translated_msg=(
                RunnableLambda(attrgetter('message'))
                | create_translator_runnable(
                    to_language=language,
                    chat_llm=create_chat_model(model, seed=seed) if language != BASE_LANGUAGE else RunnablePassthrough(),  # noqa
                    # FIXME: Conditioning by language is not good # noqa
                )
            ),
        ).with_types(input_type=MsgModel)
        | RunnableLambda(lambda dic: MsgModel(**(dic['orig'].model_dump() | {'message': dic['translated_msg']})))  # noqa
        | formatter_runnable
    )
    # create runnable
    return (
        RunnableLambda(lambda state: get_related_messsages_with_id(system_related, state))  # noqa
        | RunnableBranch(
            (
                lambda msg: msg.id not in cache,
                RunnableParallel(
                    to_cache=(
                        RunnableLambda(attrgetter('id'))
                        | RunnableLambda(cache.add)
                    ),
                    echo=(
                        RunnableLambda(attrgetter('value'))
                        | RunnableBranch(
                            *[
                                (
                                    RunnableLambda(attrgetter('name')) | RunnableLambda(name.__eq__),  # noqa
                                    formatter_runnable
                                    | create_output_runnable(
                                        output_func=kind,
                                        styler=partial(
                                            click.style,
                                            fg=color.get(name) if isinstance(color, dict) else color,  # noqa
                                        ),
                                    ),
                                )
                                for name in player_names
                            ],
                            formatter_runnable
                            | create_output_runnable(
                                output_func=kind,
                                styler=partial(
                                    click.style,
                                    fg=color.get(GAME_MASTER_NAME) if isinstance(color, dict) else color,  # noqa
                                ),
                            ),
                        )
                    ),
                )
            ),
            RunnablePassthrough(),
        ).with_types(input_type=IdentifiedModel[MsgModel]).with_config({'max_concurrency': 1}).map()  # noqa
        | RunnableLambda(lambda _: None)
    )


def create_echo_runnable(
    input_output_type: EInputOutputType,
    cli_output_level: ESystemOutputType | str,
    players: Iterable[BaseGamePlayer] = tuple(),
    players_cfg: Iterable[PlayerConfig] = tuple(),
    model: str = DEFAULT_MODEL,
    system_formatter: Callable[[MsgModel], str] | str | None = None,
    system_color: str = CLI_PROMPT_COLOR,
    player_colors: Iterable[str] = cycle(CLI_ECHO_COLORS),
    language: ELanguage = BASE_LANGUAGE,
    seed: int = -1,
) -> Runnable[StateModel, None]:
    # initialize
    player_names: list[str] = [player.name for player in players]
    player_colors_ = {player.name: color for player, color in zip(players, player_colors)}  # noqa
    # create cache
    caches: dict[str, set[str]] = (
        {player.name: {''} for player in players}
        | {GAME_MASTER_NAME: {''}}
        # NOTE: cache is initialized with a dummy value for type consistency
    )

    return (
        RunnableParallel(
            **{
                f'{DEFAULT_PLAYER_PREFIX}{i+1}': _create_echo_runnable_by_player(  # noqa
                    player=player,
                    player_config=cfg,
                    cache=caches[player.name],
                    color=player_colors_[player.name],
                    model=model,
                    input_output_type=input_output_type,
                    language=language,
                    seed=seed,
                )
                for i, (player, cfg) in enumerate(zip(players, players_cfg))
            },  # type: ignore
            **{
                GAME_MASTER_NAME: _create_echo_runnable_by_system(
                    kind=input_output_type,
                    level=cli_output_level,
                    model=model,
                    player_names=player_names,
                    cache=caches[GAME_MASTER_NAME],
                    color=player_colors_ | {GAME_MASTER_NAME: system_color},
                    language=language,
                    formatter=system_formatter,
                    seed=seed,
                ),
            },  # type: ignore
        )
        | RunnableLambda(lambda _: None)
    ).with_types(
        input_type=StateModel,
        output_type=None,
    )
