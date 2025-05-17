from typing import ClassVar

from langchain_core.runnables import RunnableLambda
import pytest

from langchain_werewolf.game_players.base import (
    BaseGamePlayerRole,
    BasePlayerSideMixin,
)
from langchain_werewolf.game_players.player_roles import (
    FortuneTeller,
    Knight,
    Villager,
    Werewolf,
)
from langchain_werewolf.game_players.registry import (
    PlayerRoleRegistry,
    PlayerSideRegistry,
)
from langchain_werewolf.game_players.player_sides import (
    VillagerSideMixin,
    WerewolfSideMixin,
)


# NOTE: The tests for the followings will be covered indirectly by confirming
#       that all other tests pass and the application runs correctly:
#       - PlayerSideRegistry
#       - PlayerRoleRegistry


class DummyCustomRoleVillager(BaseGamePlayerRole, VillagerSideMixin):
    """Dummy custom role for testing purposes."""

    role: ClassVar[str] = "DummyCustomRoleVillager"
    night_action: ClassVar[str] = "None"


class DummyCustomRoleWerewolf(BaseGamePlayerRole, WerewolfSideMixin):
    """Dummy custom role for testing purposes."""

    role: ClassVar[str] = "DummyCustomRoleWerewolf"
    night_action: ClassVar[str] = "None"


class DummpySideMixin(BasePlayerSideMixin):
    """Dummy custom side mixin for testing purposes."""

    side: ClassVar[str] = "DummyCustomSideMixin"
    victory_condition: ClassVar[str] = "None"


@pytest.mark.parametrize(
    "player_role",
    [
        Villager,
        Werewolf,
        FortuneTeller,
        Knight,
        DummyCustomRoleVillager,
        DummyCustomRoleWerewolf,
    ],
)
def test_player_role_registry_register(
    player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the registration of player roles in the PlayerRoleRegistry."""
    # execution
    assert empty_player_role_registry._registry == {}
    actual = empty_player_role_registry.register(player_role)
    # assert
    assert actual is player_role
    # NOTE: assert key is player_role.role
    assert player_role.role in empty_player_role_registry._registry
    assert empty_player_role_registry._registry[player_role.role] is player_role


@pytest.mark.parametrize(
    "invalid_player_role",
    [
        VillagerSideMixin,  # Not a subclass of BaseGamePlayerRole
        BaseGamePlayerRole,  # Not a subclass of BasePlayerSideMixin
        1,
    ],
)
def test_player_role_registry_register_invalid_plaer_role(
    invalid_player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the registration of invalid player roles in the PlayerRoleRegistry."""
    # execution
    with pytest.raises(TypeError):
        empty_player_role_registry.register(invalid_player_role)


@pytest.mark.parametrize(
    "player_role",
    [
        Villager,
        Werewolf,
        DummyCustomRoleVillager,
        DummyCustomRoleWerewolf,
    ],
)
def test_player_role_registry_unregister(
    player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the unregistration of player roles in the PlayerRoleRegistry."""
    # preparation
    assert player_role.role not in empty_player_role_registry._registry
    empty_player_role_registry._registry[player_role.role] = player_role
    assert player_role.role in empty_player_role_registry._registry
    # execution
    empty_player_role_registry.unregister(player_role)
    # assert
    assert player_role.role not in empty_player_role_registry._registry


@pytest.mark.parametrize(
    "_registry",
    [
        {
            "villager": Villager,
            "werewolf": Werewolf,
            "dummy_custom_role_villager": DummyCustomRoleVillager,
            "dummy_custom_role_werewolf": DummyCustomRoleWerewolf,
        },
        {
            "villager": Villager,
            "dummy_custom_role_werewolf": DummyCustomRoleWerewolf,
        },
    ]
)
def test_player_role_registry_get_keys(
    _registry: dict[str, type[BaseGamePlayerRole]],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the retrieval of keys from the PlayerRoleRegistry."""
    # preparation
    expected = list(_registry.keys())
    assert empty_player_role_registry._registry == {}
    empty_player_role_registry._registry = _registry
    # execute
    actual = empty_player_role_registry.get_keys()
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    "key, player_role",
    [
        (Villager.role, Villager),
        (Werewolf.role, Werewolf),
        # NOTE: the folloing cases are cases where keys does not match `role` attributes  # noqa
        ("_dummy_custom_role_villager", DummyCustomRoleVillager),
        ("_dummy_custom_role_werewolf", DummyCustomRoleWerewolf),
    ],
)
def test_player_role_registry_get_class(
    key: str,
    player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the retrieval of a class from the PlayerRoleRegistry."""
    # preparation
    assert empty_player_role_registry._registry == {}
    empty_player_role_registry._registry[key] = player_role
    # execution
    actual = empty_player_role_registry.get_class(key)
    # assert
    assert actual is player_role


def test_player_role_registry_get_class_not_found(
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the retrieval of a class from the PlayerRoleRegistry."""
    # preparation
    assert empty_player_role_registry._registry == {}
    # execution
    with pytest.raises(KeyError):
        empty_player_role_registry.get_class("not_found")


@pytest.mark.parametrize(
    "key, player_role",
    [
        (Villager.role, Villager),
        (Werewolf.role, Werewolf),
        # NOTE: the folloing cases are cases where keys does not match `role` attributes  # noqa
        ("_dummy_custom_role_villager", DummyCustomRoleVillager),
        ("_dummy_custom_role_werewolf", DummyCustomRoleWerewolf),
    ],
)
def test_player_role_registry_get_key(
    key: str,
    player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the retrieval of a key from the PlayerRoleRegistry."""

    # TODO: Add test for duplicate keys in the class registry

    # preparation
    assert empty_player_role_registry._registry == {}
    empty_player_role_registry._registry[key] = player_role
    # execution
    actual = empty_player_role_registry.get_key(player_role)
    # assert
    assert actual == key


def test_player_role_registry_get_key_not_found(
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the retrieval of a key from the PlayerRoleRegistry."""
    # preparation
    assert empty_player_role_registry._registry == {}
    # execution
    with pytest.raises(ValueError):
        empty_player_role_registry.get_key(Villager)


@pytest.mark.parametrize(
    "player_role",
    [
        Villager,
        Werewolf,
        DummyCustomRoleVillager,
        DummyCustomRoleWerewolf,
    ],
)
def test_player_role_registry_create_player(
    player_role: type[BaseGamePlayerRole],
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the creation of a player from the PlayerRoleRegistry."""
    # preparation
    name = "Alice"
    runnable = RunnableLambda(str)
    assert empty_player_role_registry._registry == {}
    empty_player_role_registry._registry[player_role.role] = player_role
    expected = player_role(
        name=name,
        runnable=runnable,
    )
    # execution
    actual = empty_player_role_registry.create_player(
        key=player_role.role,
        name=name,
        runnable=runnable,
    )
    # assert
    assert actual == expected


def test_player_role_registry_create_player_not_found(
    empty_player_role_registry: type[PlayerRoleRegistry],
) -> None:
    """Test the creation of a player from the PlayerRoleRegistry."""
    # preparation
    name = "Alice"
    runnable = RunnableLambda(str)
    assert empty_player_role_registry._registry == {}
    # execution
    with pytest.raises(KeyError):
        empty_player_role_registry.create_player(
            key="not_found",
            name=name,
            runnable=runnable,
        )


@pytest.mark.parametrize(
    "player_side",
    [
        VillagerSideMixin,
        WerewolfSideMixin,
        DummpySideMixin,
    ],
)
def test_player_side_registry_register(
    player_side: type[BasePlayerSideMixin],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the registration of player sides in the PlayerSideRegistry."""
    # execution
    assert empty_player_side_registry._registry == {}
    actual = empty_player_side_registry.register(player_side)
    # assert
    assert actual is player_side
    # NOTE: assert key is player_side.side
    assert player_side.side in empty_player_side_registry._registry
    assert (
        empty_player_side_registry._registry[player_side.side] is player_side
    )


@pytest.mark.parametrize(
    "invalid_player_side",
    [
        BaseGamePlayerRole,  # Not a subclass of BasePlayerSideMixin
        1,
    ],
)
def test_player_side_registry_register_invalid_player_side(
    invalid_player_side: type[BasePlayerSideMixin],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the registration of invalid player sides in the PlayerSideRegistry."""
    # execution
    with pytest.raises(TypeError):
        empty_player_side_registry.register(invalid_player_side)


@pytest.mark.parametrize(
    "player_side",
    [
        VillagerSideMixin,
        WerewolfSideMixin,
        DummpySideMixin,
    ],
)
def test_player_side_registry_unregister(
    player_side: type[BasePlayerSideMixin],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the unregistration of player sides in the PlayerSideRegistry."""
    # preparation
    assert player_side.side not in empty_player_side_registry._registry
    empty_player_side_registry._registry[player_side.side] = player_side
    assert player_side.side in empty_player_side_registry._registry
    # execution
    empty_player_side_registry.unregister(player_side)
    # assert
    assert player_side.side not in empty_player_side_registry._registry


@pytest.mark.parametrize(
    "_registry",
    [
        {
            "villager": VillagerSideMixin,
            "werewolf": WerewolfSideMixin,
            "dummy_custom_side_mixin": DummpySideMixin,
        },
        {
            "villager": VillagerSideMixin,
            "dummy_custom_side_mixin": DummpySideMixin,
        },
    ]
)
def test_player_side_registry_get_keys(
    _registry: dict[str, type[BasePlayerSideMixin]],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the retrieval of keys from the PlayerSideRegistry."""
    # preparation
    expected = list(_registry.keys())
    assert empty_player_side_registry._registry == {}
    empty_player_side_registry._registry = _registry
    # execute
    actual = empty_player_side_registry.get_keys()
    # assert
    assert actual == expected


@pytest.mark.parametrize(
    "key, player_side",
    [
        (VillagerSideMixin.side, VillagerSideMixin),
        (WerewolfSideMixin.side, WerewolfSideMixin),
        # NOTE: the folloing cases are cases where keys does not match `side` attributes  # noqa
        ("_dummy_custom_side_mixin", DummpySideMixin),
    ],
)
def test_player_side_registry_get_class(
    key: str,
    player_side: type[BasePlayerSideMixin],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the retrieval of a class from the PlayerSideRegistry."""
    # preparation
    assert empty_player_side_registry._registry == {}
    empty_player_side_registry._registry[key] = player_side
    # execution
    actual = empty_player_side_registry.get_class(key)
    # assert
    assert actual is player_side


def test_player_side_registry_get_class_not_found(
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the retrieval of a class from the PlayerSideRegistry."""
    # preparation
    assert empty_player_side_registry._registry == {}
    # execution
    with pytest.raises(KeyError):
        empty_player_side_registry.get_class("not_found")


@pytest.mark.parametrize(
    "key, player_side",
    [
        (VillagerSideMixin.side, VillagerSideMixin),
        (WerewolfSideMixin.side, WerewolfSideMixin),
        # NOTE: the folloing cases are cases where keys does not match `side` attributes  # noqa
        ("__dummy_custom_side_mixin", DummpySideMixin),
    ],
)
def test_player_side_registry_get_key(
    key: str,
    player_side: type[BasePlayerSideMixin],
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the retrieval of a key from the PlayerSideRegistry."""
    # preparation
    assert empty_player_side_registry._registry == {}
    empty_player_side_registry._registry[key] = player_side
    # execution
    actual = empty_player_side_registry.get_key(player_side)
    # assert
    assert actual == key


def test_player_side_registry_get_key_not_found(
    empty_player_side_registry: type[PlayerSideRegistry],
) -> None:
    """Test the retrieval of a key from the PlayerSideRegistry."""
    # preparation
    assert empty_player_side_registry._registry == {}
    # execution
    with pytest.raises(ValueError):
        empty_player_side_registry.get_key(VillagerSideMixin)
