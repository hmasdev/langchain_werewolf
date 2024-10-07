import logging
import random
import click
from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
import pydantic
from .enums import ESystemOutputType, EInputOutputType
from .game.main import create_game_graph
from .models.config import Config,  GeneralConfig
from .models.state import StateModel, MsgModel
from .setup import generate_players, create_echo_runnable
from .utils import (
    load_json,
    remove_none_values,
)

DEFAULT_CONFIG = Config(
    general=GeneralConfig(
        n_players=4,
        n_werewolves=1,
        n_knights=1,
        n_fortune_tellers=1,
        output='',
        cli_output_level=ESystemOutputType.all,
        system_interface=EInputOutputType.standard,
        system_formatter=None,
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
    n_werewolves: int = DEFAULT_GENERAL_CONFIG.n_werewolves,  # type: ignore # noqa
    n_knights: int = DEFAULT_GENERAL_CONFIG.n_knights,  # type: ignore # noqa
    n_fortune_tellers: int = DEFAULT_GENERAL_CONFIG.n_fortune_tellers,  # type: ignore # noqa
    output: str = DEFAULT_GENERAL_CONFIG.output,  # type: ignore # noqa
    cli_output_level:  ESystemOutputType | str = DEFAULT_GENERAL_CONFIG.cli_output_level,  # type: ignore # noqa
    system_interface: EInputOutputType = DEFAULT_GENERAL_CONFIG.system_interface,  # type: ignore # noqa
    system_formatter: str | None = DEFAULT_GENERAL_CONFIG.system_formatter,  # type: ignore # noqa
    config: str = '',
    seed: int = DEFAULT_GENERAL_CONFIG.seed,  # type: ignore # noqa
    model: str = DEFAULT_GENERAL_CONFIG.model,  # type: ignore # noqa
    recursion_limit: int = DEFAULT_GENERAL_CONFIG.recursion_limit,  # type: ignore # noqa
    debug: bool = False,
    verbose: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),
):
    # load config
    try:
        config_ = load_json(Config, config) if config else DEFAULT_CONFIG  # noqa
    except (pydantic.ValidationError, FileNotFoundError):
        logger.warning(f'Failed to load config: {config}')
        config_ = DEFAULT_CONFIG

    # override config
    config_used = Config(
        general=GeneralConfig(
            n_players=n_players,
            n_werewolves=n_werewolves,
            n_knights=n_knights,
            n_fortune_tellers=n_fortune_tellers,
            output=output,
            cli_output_level=cli_output_level,
            system_interface=system_interface,
            system_formatter=system_formatter,
            seed=seed,
            model=model,
            recursion_limit=recursion_limit,
            debug=debug,
            verbose=verbose,
        )
    )

    # setup
    load_dotenv()
    set_verbose(config_used.general.verbose)  # type: ignore
    set_debug(config_used.general.debug)  # type: ignore
    if config_used.general.seed >= 0:    # type: ignore
        random.seed(config_used.general.seed)

    # create players
    players = generate_players(
        config_used.general.n_players,  # type: ignore
        config_used.general.n_werewolves,  # type: ignore
        config_used.general.n_knights,  # type: ignore
        config_used.general.n_fortune_tellers,  # type: ignore
        model=config_used.general.model,
        seed=config_used.general.seed,  # type: ignore
        input_output_type=config_used.general.system_interface,  # type: ignore # noqa
        custom_players=config_used.players,
    )

    # create game workflow
    workflow = create_game_graph(
        players,
        **{
            k: remove_none_values(dic)
            for k, dic in config_.game.model_dump().items()
        },
        echo=create_echo_runnable(
            config_used.general.system_interface,  # type: ignore # noqa,
            config_used.general.cli_output_level,  # type: ignore # noqa
            players=players,
            players_cfg=config_.players,
            model=config_.general.model,  # type: ignore
            system_formatter=config_.general.system_formatter,  # type: ignore
            seed=config_.general.seed,  # type: ignore
        ),  # noqa
    )

    # run
    raw_state: dict[str, object] = workflow.invoke(
        StateModel(alive_players_names=[player.name for player in players]),
        config={"recursion_limit": config_used.general.recursion_limit},  # type: ignore  # noqa
        debug=config_.general.debug,
    )

    # save
    if config_used.general.output:
        state: StateModel = StateModel(**raw_state)  # type: ignore
        with open(config_used.general.output, 'w') as f:  # type: ignore
            f.write(state.model_dump_json(indent=4))


@click.command()
@click.option('-n', '--n-players', default=DEFAULT_GENERAL_CONFIG.n_players, help=f'The number of players. Default is {DEFAULT_GENERAL_CONFIG.n_players}.')  # noqa
@click.option('-w', '--n-werewolves', default=DEFAULT_GENERAL_CONFIG.n_werewolves, help=f'The number of werewolves. Default is {DEFAULT_GENERAL_CONFIG.n_werewolves}.')  # noqa
@click.option('-k', '--n-knights', default=DEFAULT_GENERAL_CONFIG.n_knights, help=f'The number of knights. Default is {DEFAULT_GENERAL_CONFIG.n_knights}.')  # noqa
@click.option('-f', '--n-fortune-tellers', default=DEFAULT_GENERAL_CONFIG.n_fortune_tellers, help=f'The number of fortune tellers. Default is {DEFAULT_GENERAL_CONFIG.n_fortune_tellers}.')  # noqa
@click.option('-o', '--output', default=DEFAULT_GENERAL_CONFIG.output, help=f'The output file. Defaults to "{DEFAULT_GENERAL_CONFIG.output}".')  # noqa
@click.option('-l', '--cli-output-level', default=DEFAULT_GENERAL_CONFIG.cli_output_level.name if isinstance(DEFAULT_GENERAL_CONFIG.cli_output_level, ESystemOutputType) else DEFAULT_GENERAL_CONFIG.cli_output_level, help=f'The output type of the CLI. {list(ESystemOutputType.__members__.keys())} and player names are valid. Default is All.')  # noqa
@click.option('--system-interface', default=DEFAULT_GENERAL_CONFIG.system_interface.name if isinstance(DEFAULT_GENERAL_CONFIG.system_interface, EInputOutputType) else DEFAULT_GENERAL_CONFIG.system_interface, help=f'The system interface. Default is {DEFAULT_GENERAL_CONFIG.system_interface}.')  # noqa
@click.option('--system-formatter', default=DEFAULT_GENERAL_CONFIG.system_formatter, help=f'The system formatter. The format should not include anything other than ' + ', '.join('"{'+k+'}"' for k in MsgModel.model_fields.keys()) + '.')  # noqa
@click.option('-c', '--config', default='', help='The configuration file. Defaults to "". Note that you can specify CLI arguments in this config file but the config file overwrite the CLI arguments.')  # noqa
@click.option('--seed', default=DEFAULT_GENERAL_CONFIG.seed, help=f'The random seed. Defaults to {DEFAULT_GENERAL_CONFIG.seed}.')  # noqa
@click.option('--model', default=DEFAULT_GENERAL_CONFIG.model, help=f'The model to use. Default is {DEFAULT_GENERAL_CONFIG.model}.')  # noqa
@click.option('--recursion-limit', default=DEFAULT_GENERAL_CONFIG.recursion_limit, help=f'The recursion limit. Default is {DEFAULT_GENERAL_CONFIG.recursion_limit}.')  # noqa
@click.option('--debug', is_flag=True, help='Enable debug mode.')
@click.option('--verbose', is_flag=True, help='Enable verbose mode.')
def cli(
    n_players: int = DEFAULT_GENERAL_CONFIG.n_players,  # type: ignore # noqa
    n_werewolves: int = DEFAULT_GENERAL_CONFIG.n_werewolves,  # type: ignore # noqa
    n_knights: int = DEFAULT_GENERAL_CONFIG.n_knights,  # type: ignore # noqa
    n_fortune_tellers: int = DEFAULT_GENERAL_CONFIG.n_fortune_tellers,  # type: ignore # noqa
    output: str = DEFAULT_GENERAL_CONFIG.output,  # type: ignore # noqa
    cli_output_level:  str = DEFAULT_GENERAL_CONFIG.cli_output_level.name,  # type: ignore # noqa
    system_interface: str = DEFAULT_GENERAL_CONFIG.system_interface.name,  # type: ignore # noqa
    config: str = '',  # type: ignore # noqa
    seed: int = DEFAULT_GENERAL_CONFIG.seed,  # type: ignore # noqa
    model: str = DEFAULT_GENERAL_CONFIG.model,  # type: ignore # noqa
    recursion_limit: int = DEFAULT_GENERAL_CONFIG.recursion_limit,  # type: ignore # noqa
    debug: bool = False,  # type: ignore # noqa
    verbose: bool = False,  # type: ignore # noqa
    logger: logging.Logger = logging.getLogger(__name__),  # type: ignore # noqa,
):
    if hasattr(ESystemOutputType, cli_output_level):
        cli_output_level = ESystemOutputType(cli_output_level)  # type: ignore
    if hasattr(EInputOutputType, system_interface):
        system_interface = EInputOutputType(system_interface)  # type: ignore
    else:
        raise ValueError(f'Invalid system interface: {system_interface}. Valid values are {list(EInputOutputType.__members__.keys())}.')  # noqa
    main(
        n_players=n_players,
        n_werewolves=n_werewolves,
        n_knights=n_knights,
        n_fortune_tellers=n_fortune_tellers,
        output=output,
        cli_output_level=cli_output_level,
        system_interface=system_interface,  # type: ignore
        config=config,
        seed=seed,
        model=model,
        recursion_limit=recursion_limit,
        debug=debug,
        verbose=verbose,
        logger=logger,
    )
