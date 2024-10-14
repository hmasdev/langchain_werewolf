from glob import glob
import json
import os
import subprocess
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain_werewolf.main import DEFAULT_CONFIG
from langchain_werewolf.game.main import create_game_graph
from langchain_werewolf.setup import generate_players

# load env
load_dotenv()

# const
WORKDIR: str = '.'
EXAMPLES_DIR: str = 'examples'
EXAMPLES_CONFIG_DIR: str = os.path.join(EXAMPLES_DIR, 'config')
README_TEMPLATE_NAME: str = 'README.md.j2'

# vars
readme = Environment(loader=FileSystemLoader(WORKDIR)).get_template(README_TEMPLATE_NAME)  # noqa
help: str = subprocess.run(
    ['python', '-m', 'langchain_werewolf', '--help'],
    capture_output=True,
    text=True,
).stdout
example_configs: list[dict[str, str | int | dict | list]] = [
    json.load(open(path))  # noqa
    for path in sorted(glob(os.path.join(EXAMPLES_CONFIG_DIR, '*.json')))
]

# render
print(readme.render(
    help=help,
    example_configs=example_configs,
))

# create graphs
graph = create_game_graph(
    generate_players(
        DEFAULT_CONFIG.general.n_players,
        DEFAULT_CONFIG.general.n_werewolves,
        DEFAULT_CONFIG.general.n_knights,
        DEFAULT_CONFIG.general.n_fortune_tellers,
        custom_players=DEFAULT_CONFIG.players,
        seed=DEFAULT_CONFIG.general.seed,
    ),
)
# update graphs
graph.get_graph(xray=False).draw_png('pics/langchain_werewolf_game_graph_simple.png')  # noqa
graph.get_graph(xray=True).draw_png('pics/langchain_werewolf_game_graph.png')  # noqa
