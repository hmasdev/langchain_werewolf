from itertools import cycle
import logging
import random
from typing import Any, Callable, Iterable
import click
from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
import pydantic
from .const import BASE_LANGUAGE, CLI_PROMPT_COLOR, CLI_ECHO_COLORS
from .enums import ESystemOutputType, EInputOutputType, ELanguage
from .game.main import create_game_graph
from .game_players import (
    PlayerRoleRegistry,
    PlayerSideRegistry,
    WEREWOLF_ROLE,
)
from .models.config import Config,  GeneralConfig
from .models.state import StateModel, MsgModel
from .setup import generate_players, create_echo_runnable
from .utils import (
    load_json,
    remove_none_values,
)

# Initialize
PlayerRoleRegistry.initialize()
PlayerSideRegistry.initialize()

DEFAULT_CONFIG = Config(
    general=GeneralConfig(
        n_players=4,
        n_players_by_role={
            WEREWOLF_ROLE: 1,
        },
        output='',
        system_output_level=ESystemOutputType.all,
        system_output_interface=EInputOutputType.standard,
        system_input_interface=EInputOutputType.standard,
        system_language=BASE_LANGUAGE,
        system_formatter=None,
        system_font_color=CLI_PROMPT_COLOR,
        player_font_colors=cycle(CLI_ECHO_COLORS),
        seed=-1,
        model='gpt-4o-mini',
        recursion_limit=1000,
        debug=False,
        verbose=False,
    )
)
DEFAULT_GENERAL_CONFIG = DEFAULT_CONFIG.general


def main(
    n_players: int = DEFAULT_GENERAL_CONFIG.n_players,  # type: ignore # noqa
    n_players_by_role: dict[str, int] = DEFAULT_GENERAL_CONFIG.n_players_by_role,  # type: ignore # noqa
    output: str = DEFAULT_GENERAL_CONFIG.output,  # type: ignore # noqa
    system_output_level:  ESystemOutputType | str = DEFAULT_GENERAL_CONFIG.system_output_level,  # type: ignore # noqa
    system_output_interface: Callable[[str], None] | EInputOutputType = DEFAULT_GENERAL_CONFIG.system_output_interface,  # type: ignore # noqa
    system_input_interface: Callable[[str], Any] | EInputOutputType = DEFAULT_GENERAL_CONFIG.system_input_interface,  # type: ignore # noqa
    system_language: ELanguage | None = DEFAULT_GENERAL_CONFIG.system_language,  # type: ignore # noqa
    system_formatter: str | None = DEFAULT_GENERAL_CONFIG.system_formatter,  # type: ignore # noqa
    system_font_color: str | None = DEFAULT_GENERAL_CONFIG.system_font_color,  # type: ignore # noqa
    player_font_colors: Iterable[str] | str | None = DEFAULT_GENERAL_CONFIG.player_font_colors,  # type: ignore # noqa
    config: Config | str | None = None,
    seed: int = DEFAULT_GENERAL_CONFIG.seed,  # type: ignore # noqa
    model: str = DEFAULT_GENERAL_CONFIG.model,  # type: ignore # noqa
    recursion_limit: int = DEFAULT_GENERAL_CONFIG.recursion_limit,  # type: ignore # noqa
    debug: bool = False,
    verbose: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),
) -> StateModel:
    # load config
    if isinstance(config, str) and config != '':
        try:
            logger.info(f"Load config from a file: {config}.")
            config = load_json(Config, config)
        except (pydantic.ValidationError, FileNotFoundError) as e:
            raise Exception(f'Failed to load config from a file: {config}. Check the file existence and its format.') from e  # noqa
    elif config == "":
        config = None
    if not isinstance(config, Config) and config is not None:
        raise ValueError(f'Invalid config type: {type(config)}. Expected Config.')  # noqa

    # override config
    # NOTE: Priority order: config > CLI arguments > DEFAULT_CONFIG
    config_used = Config(
        general=GeneralConfig(
            n_players=config.general.n_players if (config is not None and config.general.n_players is not None) else n_players,  # noqa
            n_players_by_role=config.general.n_players_by_role if (config is not None and config.general.n_players_by_role is not None) else n_players_by_role,  # noqa
            output=config.general.output if (config is not None and config.general.output is not None) else output,  # noqa
            system_output_level=config.general.system_output_level if (config is not None and config.general.system_output_level is not None) else system_output_level,  # noqa
            system_input_interface=config.general.system_input_interface if (config is not None and config.general.system_input_interface is not None) else system_input_interface,  # noqa
            system_output_interface=config.general.system_output_interface if (config is not None and config.general.system_output_interface is not None) else system_output_interface,  # noqa
            system_language=config.general.system_language if (config is not None and config.general.system_language is not None) else system_language,  # noqa
            system_formatter=config.general.system_formatter if (config is not None and config.general.system_formatter is not None) else system_formatter,  # noqa
            system_font_color=config.general.system_font_color if (config is not None and config.general.system_font_color is not None) else system_font_color,  # noqa
            player_font_colors=config.general.player_font_colors if (config is not None and config.general.player_font_colors is not None) else player_font_colors,  # noqa
            seed=config.general.seed if (config is not None and config.general.seed is not None) else seed,  # noqa
            model=config.general.model if (config is not None and config.general.model is not None) else model,  # noqa
            recursion_limit=config.general.recursion_limit if (config is not None and config.general.recursion_limit is not None) else recursion_limit,  # noqa
            debug=config.general.debug if (config is not None and config.general.debug is not None) else debug,  # noqa
            verbose=config.general.verbose if (config is not None and config.general.verbose is not None) else verbose,  # noqa
        ),
        players=config.players if (config is not None and config.players is not None) else DEFAULT_CONFIG.players,  # noqa
        game=config.game if (config is not None and config.game is not None) else DEFAULT_CONFIG.game,  # noqa
    )

    # setup
    load_dotenv(override=True)
    set_verbose(config_used.general.verbose)  # type: ignore
    set_debug(config_used.general.debug)  # type: ignore
    if config_used.general.seed >= 0:    # type: ignore
        random.seed(config_used.general.seed)

    # create players
    players = generate_players(
        config_used.general.n_players,  # type: ignore
        config_used.general.n_players_by_role,
        model=config_used.general.model,
        seed=config_used.general.seed,  # type: ignore
        player_input_interface=config_used.general.system_input_interface,  # type: ignore # noqa
        custom_players=config_used.players,
    )

    # create game workflow
    workflow = create_game_graph(
        players,
        **{
            k: remove_none_values(dic)
            for k, dic in config_used.game.model_dump().items()
        },
        echo=create_echo_runnable(
            config_used.general.system_output_interface,  # type: ignore # noqa,
            config_used.general.system_output_level,  # type: ignore # noqa
            players=players,
            model=config_used.general.model,  # type: ignore
            system_formatter=config_used.general.system_formatter,  # type: ignore # noqa
            system_color=config_used.general.system_font_color,  # type: ignore # noqa
            player_colors=config_used.general.player_font_colors,  # type: ignore # noqa
            seed=config_used.general.seed,  # type: ignore
            language=config_used.general.system_language,  # type: ignore
        ),
    )

    # run
    raw_state: dict[str, object] = workflow.invoke(
        StateModel(alive_players_names=[player.name for player in players]),
        config={"recursion_limit": config_used.general.recursion_limit},  # type: ignore  # noqa
        debug=config_used.general.debug,
    )
    state: StateModel = StateModel(**raw_state)  # type: ignore

    # save
    if config_used.general.output:
        with open(config_used.general.output, 'w') as f:  # type: ignore
            f.write(state.model_dump_json(indent=4))

    return state


def attach_n_players_by_role_options(
    cli: Callable[..., None],
) -> Callable[..., None]:

    roles = PlayerRoleRegistry.get_keys()
    roles = [role for role in roles if role != WEREWOLF_ROLE] + [WEREWOLF_ROLE]

    for role in roles:
        cli = click.option(
            f'--n-{role}',
            default=DEFAULT_GENERAL_CONFIG.n_players_by_role.get(role, 0),
            help=f'The number of {role}s. Default is {DEFAULT_GENERAL_CONFIG.n_players_by_role.get(role, 0)}.',  # noqa
        )(cli)

    return cli


@click.command()
@click.option('-n', '--n-players', default=DEFAULT_GENERAL_CONFIG.n_players, help=f'The number of players. Default is {DEFAULT_GENERAL_CONFIG.n_players}.')  # noqa
@attach_n_players_by_role_options
@click.option('-o', '--output', default=DEFAULT_GENERAL_CONFIG.output, help=f'The output file. Defaults to "{DEFAULT_GENERAL_CONFIG.output}".')  # noqa
@click.option('-l', '--system-output-level', default=DEFAULT_GENERAL_CONFIG.system_output_level.name if isinstance(DEFAULT_GENERAL_CONFIG.system_output_level, ESystemOutputType) else DEFAULT_GENERAL_CONFIG.system_output_level, help=f'The output type of the CLI. {list(ESystemOutputType.__members__.keys())} and player names are valid. Default is All.')  # noqa
@click.option('--system-output-interface', default=DEFAULT_GENERAL_CONFIG.system_output_interface.name if isinstance(DEFAULT_GENERAL_CONFIG.system_output_interface, EInputOutputType) else DEFAULT_GENERAL_CONFIG.system_output_interface, help=f'The system interface. Default is {DEFAULT_GENERAL_CONFIG.system_output_interface}.')  # noqa
@click.option('--system-input-interface', default=DEFAULT_GENERAL_CONFIG.system_input_interface.name if isinstance(DEFAULT_GENERAL_CONFIG.system_input_interface, EInputOutputType) else DEFAULT_GENERAL_CONFIG.system_input_interface, help=f'The system interface. Default is {DEFAULT_GENERAL_CONFIG.system_input_interface}.')  # noqa
@click.option('--system-formatter', default=DEFAULT_GENERAL_CONFIG.system_formatter, help=f'The system formatter. The format should not include anything other than ' + ', '.join('"{'+k+'}"' for k in MsgModel.model_fields.keys()) + '.')  # noqa
@click.option('-c', '--config', default='', help='The configuration file. Defaults to "". Note that you can specify CLI arguments in this config file but the config file overwrite the CLI arguments.')  # noqa
@click.option('--seed', default=DEFAULT_GENERAL_CONFIG.seed, help=f'The random seed. Defaults to {DEFAULT_GENERAL_CONFIG.seed}.')  # noqa
@click.option('--model', default=DEFAULT_GENERAL_CONFIG.model, help=f'The model to use. Default is {DEFAULT_GENERAL_CONFIG.model}.')  # noqa
@click.option('--recursion-limit', default=DEFAULT_GENERAL_CONFIG.recursion_limit, help=f'The recursion limit. Default is {DEFAULT_GENERAL_CONFIG.recursion_limit}.')  # noqa
@click.option('--debug', is_flag=True, help='Enable debug mode.')
@click.option('--verbose', is_flag=True, help='Enable verbose mode.')
def cli(
    *,
    n_players: int = DEFAULT_GENERAL_CONFIG.n_players,  # type: ignore # noqa
    output: str = DEFAULT_GENERAL_CONFIG.output,  # type: ignore # noqa
    system_output_level:  str = DEFAULT_GENERAL_CONFIG.system_output_level.name,  # type: ignore # noqa
    system_output_interface: str = DEFAULT_GENERAL_CONFIG.system_output_interface.name,  # type: ignore # noqa
    system_input_interface: str = DEFAULT_GENERAL_CONFIG.system_input_interface.name,  # type: ignore # noqa
    system_formatter: str = DEFAULT_GENERAL_CONFIG.system_formatter,  # type: ignore # noqa
    config: str = '',  # type: ignore # noqa
    seed: int = DEFAULT_GENERAL_CONFIG.seed,  # type: ignore # noqa
    model: str = DEFAULT_GENERAL_CONFIG.model,  # type: ignore # noqa
    recursion_limit: int = DEFAULT_GENERAL_CONFIG.recursion_limit,  # type: ignore # noqa
    debug: bool = False,  # type: ignore # noqa
    verbose: bool = False,  # type: ignore # noqa
    logger: logging.Logger = logging.getLogger(__name__),  # type: ignore # noqa,
    **kwargs,
):
    if hasattr(ESystemOutputType, system_output_level):
        system_output_level = ESystemOutputType(system_output_level)  # type: ignore # noqa
    if hasattr(EInputOutputType, system_output_interface):
        system_output_interface = EInputOutputType(system_output_interface)  # type: ignore  # noqa
    else:
        raise ValueError(f'Invalid system interface: {system_output_interface}. Valid values are {list(EInputOutputType.__members__.keys())}.')  # noqa
    main(
        n_players=n_players,
        n_players_by_role={k.replace("n_", ""): int(v) for k, v in kwargs.items()},  # noqa
        output=output,
        system_output_level=system_output_level,
        system_output_interface=system_output_interface,  # type: ignore
        system_input_interface=EInputOutputType(system_input_interface),
        system_formatter=system_formatter,
        config=config,
        seed=seed,
        model=model,
        recursion_limit=recursion_limit,
        debug=debug,
        verbose=verbose,
        logger=logger,
    )
