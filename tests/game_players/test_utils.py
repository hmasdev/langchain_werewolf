from langchain_core.runnables import RunnableLambda
import pytest

from langchain_werewolf.game_players.base import (
    BaseGamePlayer,
    BaseGamePlayerRole,
    BasePlayerSideMixin,
)
from langchain_werewolf.game_players.registry import PlayerRoleRegistry
from langchain_werewolf.game_players.player_roles import Villager, Werewolf
from langchain_werewolf.game_players.player_sides import (
    VillagerSideMixin,
    WerewolfSideMixin,
)
from langchain_werewolf.game_players.utils import (
    find_player_by_name,
    find_players_by_role,
    find_players_by_side,
    is_player_with_role,
    is_player_with_side,
    is_werewolf_side,
    is_werewolf_role,
    is_valid_game_player,
)


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (Werewolf(name="Bob", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (BaseGamePlayer(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BaseGamePlayerRole(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
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
        (Villager(name="Alice", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (Werewolf(name="Bob", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (BaseGamePlayer(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BaseGamePlayerRole(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), True),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
    ]
)
def test_is_werewolf_side(
    player: BasePlayerSideMixin,
    expected: bool,
) -> None:
    assert is_werewolf_side(player) == expected


@pytest.mark.parametrize(
    "player,expected",
    [
        (Villager(name="Alice", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (Werewolf(name="Bob", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (BaseGamePlayer(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BaseGamePlayerRole(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BasePlayerSideMixin(), True),
        (VillagerSideMixin(), True),
        (WerewolfSideMixin(), True),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
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
        (Villager(name="Alice", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (Werewolf(name="Bob", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (BaseGamePlayer(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BaseGamePlayerRole(name="Charlie",
         runnable=RunnableLambda(lambda _: "hello")), True),
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
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
        (Villager(name="Alice", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (Werewolf(name="Bob", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
        (BaseGamePlayer(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BaseGamePlayerRole(name="Charlie", runnable=RunnableLambda(lambda _: "hello")), False),  # noqa
        (BasePlayerSideMixin(), False),
        (VillagerSideMixin(), False),
        (WerewolfSideMixin(), False),
        (type("InheritedWerewolf", (Werewolf,), {})(name="Dave", runnable=RunnableLambda(lambda _: "hello")), True),  # noqa
    ]
)
def test_is_valid_game_player(
    player: BaseGamePlayer,
    expected: bool,
) -> None:
    assert is_valid_game_player(player) == expected


@pytest.mark.parametrize(
    'name, players, expected',
    [
        (
            'Alice',
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
            ],
            PlayerRoleRegistry.create_player(
                key=Villager.role,
                name='Alice',
                runnable=RunnableLambda(str),
            ),
        ),
        (
            'Bob',
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
            ],
            PlayerRoleRegistry.create_player(
                key=Werewolf.role,
                name='Bob',
                runnable=RunnableLambda(str),
            ),
        ),
    ],
)
def test_find_player_by_name(
    name: str,
    players: list[BaseGamePlayer],
    expected: BaseGamePlayer,
):
    assert find_player_by_name(name, players) == expected


def test_find_player_by_name_not_found():
    with pytest.raises(ValueError):
        find_player_by_name('Alice', [])


def test_find_player_by_name_not_unique():
    with pytest.raises(ValueError):
        find_player_by_name(
            'Alice',
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
            ] * 3,
        )


@pytest.mark.parametrize(
    'role, players, expected',
    [
        (
            Villager.role,
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
        ),
        (
            Werewolf.role,
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
            [
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
            ],
        ),
    ],
)
def test_find_players_by_role(
    role: str,
    players: list[BaseGamePlayer],
    expected: list[BaseGamePlayer],
):
    assert find_players_by_role(role, players) == expected


def test_find_players_by_role_not_found():
    with pytest.raises(ValueError):
        find_players_by_role(Villager.role, [])


@pytest.mark.parametrize(
    'side, players, expected',
    [
        (
            Villager.side,
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
        ),
        (
            Werewolf.side,
            [
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Alice',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
                PlayerRoleRegistry.create_player(
                    key=Villager.role,
                    name='Charlie',
                    runnable=RunnableLambda(str),
                ),
            ],
            [
                PlayerRoleRegistry.create_player(
                    key=Werewolf.role,
                    name='Bob',
                    runnable=RunnableLambda(str),
                ),
            ],
        ),
    ],
)
def test_find_players_by_side(
    side: str,
    players: list[BaseGamePlayer],
    expected: list[BaseGamePlayer],
):
    assert find_players_by_side(side, players) == expected


def test_find_players_by_side_not_found():
    with pytest.raises(ValueError):
        find_players_by_side(Villager.side, [])
