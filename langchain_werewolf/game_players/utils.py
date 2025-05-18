from typing import Iterable, TypeGuard

from .base import BaseGamePlayer, BaseGamePlayerRole, BasePlayerSideMixin
from .const import WEREWOLF_ROLE, WEREWOLF_SIDE
from .registry import PlayerRoleRegistry, PlayerSideRegistry
from ..utils import assert_not_empty_deco


def is_werewolf_role(
    player: BaseGamePlayer,
) -> bool:
    """Check if the player is a werewolf role

    Args:
        player (BaseGamePlayer):
            an instance of a subclass of BaseGamePlayer representing a player.

    Returns:
        bool:
            True if the player is a werewolf role, that is, an instance of
            Werewolf, False otherwise
    """
    return isinstance(player, PlayerRoleRegistry.get_class(WEREWOLF_ROLE))


def is_werewolf_side(
    player: BasePlayerSideMixin,
) -> bool:
    """Check if the player is a werewolf side

    Args:
        player (BasePlayerSideMixin):
            an instance of a subclass of BasePlayerSideMixin representing
            a player side.

    Returns:
        bool:
            True if the player is a werewolf side, that is, an instance of
            Werewolf, False otherwise
    """
    return isinstance(player, PlayerSideRegistry.get_class(WEREWOLF_SIDE))


def is_player_with_side(
    player: BaseGamePlayer,
) -> TypeGuard[BasePlayerSideMixin]:
    """Check if the player is associated with a valid player side

    Args:
        player (BaseGamePlayer):
            an instance of a subclass of BaseGamePlayer representing a player.

    Returns:
        bool:
            True if the player is associated with a valid player side,
            that is, an instance of BasePlayerSideMixin, False otherwise
    """
    return isinstance(player, BasePlayerSideMixin)


def is_player_with_role(
    player: BaseGamePlayer,
) -> TypeGuard[BaseGamePlayerRole]:
    """Check if the player is associated with a valid player role

    Args:
        player (BaseGamePlayer):
            an instance of a subclass of BaseGamePlayer representing a player.

    Returns:
        bool:
            True if the player is associated with a valid player role,
            that is, an instance of BaseGamePlayerRole, False otherwise
    """  # noqa
    return isinstance(player, BaseGamePlayerRole)


def is_valid_game_player(
    player: BaseGamePlayer,
) -> bool:
    """Check if the player is a valid game player

    Args:
        player (BaseGamePlayer):
            an instance of a subclass of BaseGamePlayer representing a player.

    Returns:
        bool:
            True if the player is a valid game player, that is, an instance
            inheriting from both BasePlayerSideMixin and BaseGamePlayerRole,
            False otherwise
    """  # noqa
    return is_player_with_role(player) and is_player_with_side(player)


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
        KeyError: if the role is not registered
        ValueError: if there are no players with the role
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
        ValueError: if there are no players with the side
    """
    side_cls = PlayerSideRegistry.get_class(side)
    return [
        player
        for player in players
        if isinstance(player, side_cls)
    ]
