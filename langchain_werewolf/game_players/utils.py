from typing import TypeGuard

from .base import BaseGamePlayer, BaseGamePlayerRole, BasePlayerSideMixin
from .const import WEREWOLF_ROLE, WEREWOLF_SIDE
from .registry import PlayerRoleRegistry, PlayerSideRegistry


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

    Reurns:
        bool:
            True if the player is a valid game player, that is, an instance
            inheriting from both BasePlayerSideMixin and BaseGamePlayerRole,
            False otherwise
    """  # noqa
    return is_player_with_role(player) and is_player_with_side(player)
