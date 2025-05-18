from typing import Generator

import pytest

from langchain_werewolf.game_players.registry import (
    PlayerRoleRegistry,
    PlayerSideRegistry,
)


@pytest.fixture(autouse=True, scope="session")
def registry_initialized() -> Generator[tuple[type[PlayerRoleRegistry], type[PlayerSideRegistry]], None, None]:  # noqa
    """Fixture to initialize the PlayerRoleRegistry and PlayerSideRegistry."""
    PlayerRoleRegistry.initialize()
    PlayerSideRegistry.initialize()
    yield (
        PlayerRoleRegistry,
        PlayerSideRegistry,
    )


@pytest.fixture(autouse=False, scope="function")
def empty_player_role_registry() -> Generator[type[PlayerRoleRegistry], None, None]:  # noqa
    """Fixture to clear the PlayerRoleRegistry."""
    cache = PlayerRoleRegistry._registry
    PlayerRoleRegistry._registry = {}
    yield PlayerRoleRegistry
    PlayerRoleRegistry._registry = cache


@pytest.fixture(autouse=False, scope="function")
def empty_player_side_registry() -> Generator[type[PlayerSideRegistry], None, None]:  # noqa
    """Fixture to clear the PlayerSideRegistry."""
    cache = PlayerSideRegistry._registry
    PlayerSideRegistry._registry = {}
    yield PlayerSideRegistry
    PlayerSideRegistry._registry = cache
