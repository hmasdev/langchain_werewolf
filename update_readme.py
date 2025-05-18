from glob import glob
import json
import os
import subprocess
from unittest.mock import patch, Mock
from jinja2 import Environment, FileSystemLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_werewolf.main import DEFAULT_CONFIG
from langchain_werewolf.game.main import create_game_graph
from langchain_werewolf.setup import generate_players

# const
WORKDIR: str = '.'
EXAMPLES_DIR: str = 'examples'
PACKAGE_DIR: str = 'langchain_werewolf'
EXAMPLES_CONFIG_DIR: str = os.path.join(EXAMPLES_DIR, 'config')
ROLE_IMPLEMENTATION_DIR: str = os.path.join(PACKAGE_DIR, 'game_players', 'player_roles')  # noqa
README_TEMPLATE_NAME: str = 'README.md.j2'

# vars
readme = Environment(loader=FileSystemLoader(WORKDIR)).get_template(README_TEMPLATE_NAME)  # noqa
help: str = subprocess.run(
    ['python', '-m', 'langchain_werewolf', '--help'],
    capture_output=True,
    text=True,
).stdout
example_configs: list[dict[str, str | int | dict | list]] = [
    json.dumps(json.load(open(path)), indent=4)  # noqa
    for path in sorted(glob(os.path.join(EXAMPLES_CONFIG_DIR, '*.json')))
]
role_implementations: dict[str, str] = {
    os.path.basename(path).replace('.py', ''): open(path).read()
    for path in glob(os.path.join(ROLE_IMPLEMENTATION_DIR, '*.py'))
}

# render
print(readme.render(
    help=help,
    configs=example_configs,
    role_implementations=role_implementations,
))

with patch(
    'langchain_werewolf.setup._generate_base_runnable',
    Mock(return_value=RunnablePassthrough().with_types(input_type=str, output_type=str)),  # noqa
):
    # TODO: remove patch
    # create graphs
    graph = create_game_graph(
        generate_players(
            DEFAULT_CONFIG.general.n_players,
            DEFAULT_CONFIG.general.n_players_by_role,
            custom_players=DEFAULT_CONFIG.players,
            seed=DEFAULT_CONFIG.general.seed,
        ),
    )
    # update graphs
    graph.get_graph(xray=False).draw_png('pics/langchain_werewolf_game_graph_simple.png')  # noqa
    graph.get_graph(xray=True).draw_png('pics/langchain_werewolf_game_graph.png')  # noqa
