import pytest

from langchain_werewolf.game_players.base import (
    BaseGamePlayer,
    BaseGamePlayerRole,
    BasePlayerSideMixin,
)
from langchain_werewolf.game_players.player_roles import Villager, Werewolf
from langchain_werewolf.game_players.player_sides import (
    VillagerSideMixin,
    WerewolfSideMixin,
)
from langchain_werewolf.game_players.utils import (
    is_player_with_role,
    is_player_with_side,
    is_werewolf_side,
    is_werewolf_role,
    is_valid_game_player,
)


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=lambda _: "hello"), False),
        (Werewolf(name="Bob", runnable=lambda _: "hello"), True),
        (BaseGamePlayer(name="Charlie", runnable=lambda _: "hello"), False),
        (BaseGamePlayerRole(name="Charlie", runnable=lambda _: "hello"), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=lambda _: "hello"), True),  # noqa
    ]
)
def test_is_werewolf_role(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_werewolf_role(player) == expected


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=lambda _: "hello"), False),
        (Werewolf(name="Bob", runnable=lambda _: "hello"), True),
        (BaseGamePlayer(name="Charlie", runnable=lambda _: "hello"), False),
        (BaseGamePlayerRole(name="Charlie", runnable=lambda _: "hello"), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), True),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=lambda _: "hello"), True),  # noqa
    ]
)
def test_is_werewolf_side(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_werewolf_side(player) == expected


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=lambda _: "hello"), True),
        (Werewolf(name="Bob", runnable=lambda _: "hello"), True),
        (BaseGamePlayer(name="Charlie", runnable=lambda _: "hello"), False),
        (BaseGamePlayerRole(name="Charlie", runnable=lambda _: "hello"), False),  # noqa
        (BasePlayerSideMixin(), True),
        (VillagerSideMixin(), True),
        (WerewolfSideMixin(), True),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=lambda _: "hello"), True),  # noqa
    ]
)
def test_is_player_with_side(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_player_with_side(player) == expected


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=lambda _: "hello"), True),
        (Werewolf(name="Bob", runnable=lambda _: "hello"), True),
        (BaseGamePlayer(name="Charlie", runnable=lambda _: "hello"), False),
        (BaseGamePlayerRole(name="Charlie", runnable=lambda _: "hello"), True),
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=lambda _: "hello"), True),  # noqa
    ]
)
def test_is_player_with_role(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_player_with_role(player) == expected


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=lambda _: "hello"), True),
        (Werewolf(name="Bob", runnable=lambda _: "hello"), True),
        (BaseGamePlayer(name="Charlie", runnable=lambda _: "hello"), False),
        (BaseGamePlayerRole(name="Charlie", runnable=lambda _: "hello"), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=lambda _: "hello"), True),  # noqa
    ]
)
def test_is_valid_game_player(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_valid_game_player(player) == expected
