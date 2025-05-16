from copy import deepcopy
from functools import wraps
import random
from typing import Callable, Generator, Iterable, Sized, TypeVar

from pydantic import BaseModel
import pydantic_core

from .game_players import (
    BaseGamePlayer,
    PlayerRoleRegistry,
    PlayerSideRegistry,
)

T = TypeVar('T')
PydanticBaseModel = TypeVar('PydanticBaseModel', bound=BaseModel)


def assert_not_empty_deco(func: Callable[..., Sized]) -> Callable[..., Sized]:
    """A decorator to assert that the input is not empty

    Args:
        func (Callable[..., Sized]): the function to be decorated

    Returns:
        Callable[..., Sized]: the decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if len(result) == 0:
            raise ValueError('The input is empty.')
        return func(*args, **kwargs)
    return wrapper


def find_player_by_name(
    name: str,
    players: Iterable[BaseGamePlayer],
) -> BaseGamePlayer:
    """Find a player by name

    Args:
        name (str): the name of the player
        players (Iterable[BaseGamePlayer]): the list of players

    Raises:
        ValueError: player not found
        ValueError: player name is not unique

    Returns:
        BaseGamePlayer: the player with the name
    """
    players = list(filter(lambda x: x.name == name, players))
    if len(players) == 0:
        raise ValueError(f'The name {name} is not found.')
    if len(players) > 1:
        raise ValueError(f'The name {name} is not unique.')
    return players[0]


@assert_not_empty_deco
def find_players_by_role(
    role: str,
    players: Iterable[BaseGamePlayer],
) -> list[BaseGamePlayer]:
    """Find players by role

    Args:
        role (str): the role to be found
        players (Iterable[BaseGamePlayer]): the list of players

    Returns:
        list[BaseGamePlayer]: players with the role

    Raises:
        KeyError: the role is not registered
    """
    role_cls = PlayerRoleRegistry.get_class(role)
    return [
        player
        for player in players
        if isinstance(player, role_cls)
    ]


@assert_not_empty_deco
def find_players_by_side(
    side: str,
    players: Iterable[BaseGamePlayer],
) -> list[BaseGamePlayer]:
    """Find players by side

    Args:
        side (str): the side to be found
        players (Iterable[BaseGamePlayer]): the list of players

    Returns:
        list[BaseGamePlayer]: players with the side

    Raises:
        KeyError: the side is not registered
    """
    side_cls = PlayerSideRegistry.get_class(side)
    return [
        player
        for player in players
        if isinstance(player, side_cls)
    ]


def consecutive_string_generator(
    prefix: str,
    start: int = 0,
    step: int = 1,
) -> Generator[str, None, None]:
    """a generator to generate strings with a prefix and a number.

    Args:
        prefix (str): prefix of the string
        start (int, optional): start number. Defaults to 0.
        step (int, optional): step of the number. Defaults to 1.

    Yields:
        Generator[str, None, None]: the generated string
            format: f'{prefix}{number}'
    """
    idx = start
    while True:
        yield f'{prefix}{idx}'
        idx += step


def random_permutated_infinite_generator(
    objects: Iterable[T],
) -> Generator[T, None, None]:
    """a generator to generate random permutated objects infinitely

    Args:
        objects (Iterable[T]): the objects to be permutated

    Yields:
        Generator[T, None, None]: infinite generator with random permutated objects
    """  # noqa
    lst_objects = deepcopy(list(objects))
    while True:
        random.shuffle(lst_objects)
        for obj in lst_objects:
            yield obj


def remove_none_values(d: dict[str, T | None]) -> dict[str, T]:
    """Remove the None values from the dictionary

    Args:
        d (dict[str, T  |  None]): the dictionary from which to remove the None values

    Returns:
        dict[str, T]: the dictionary without the None values
    """  # noqa
    return {k: v for k, v in d.items() if v is not None}


def load_json(
    Model: type[PydanticBaseModel],
    path: str,
) -> PydanticBaseModel:
    """Load a pydantic model from a json file

    Args:
        Model (type[PydanticBaseModel]): the pydantic model
        path (str): the path to the file

    Raises:
        FileNotFoundError: the file is not found
        pydantic.ValidationError: the loaded json is invalid

    Returns:
        PydanticBaseModel: the loaded pydantic model
    """  # noqa
    with open(path, 'r') as f:
        contents = f.read()
    return Model.model_validate(
        pydantic_core.from_json(contents, allow_partial=True),
    )
