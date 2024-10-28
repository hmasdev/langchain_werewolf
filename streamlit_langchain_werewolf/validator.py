from collections import Counter
from typing import Callable
from langchain_werewolf.enums import ERole
from langchain_werewolf.models.config import PlayerConfig
from streamlit_langchain_werewolf.const import HUMAN


def validate_number_of_players(
    n_players: int,
    n_werewolves: int,
    n_knights: int,
    n_fortune_tellers: int,
    *,
    alert_func: Callable[[str], None] | None = None
) -> bool | tuple[bool, str]:
    alert_func = alert_func or (lambda _: None)
    result: bool = True
    if n_players < n_werewolves + n_knights + n_fortune_tellers:
        alert_func("Validati\onError: the number of players must be >= the sum of the number of werewolves, knights, and fortune tellers.")  # noqa
        result = False
    if n_werewolves < 1:
        alert_func("ValidationError: the number of werewolves must be >= 1.")  # noqa
        result = False
    if n_knights < 0:
        alert_func("ValidationError: the number of knights must be >= 0.")  # noqa
        result = False
    if n_fortune_tellers < 0:
        alert_func("ValidationError: the number of fortune tellers must be >= 0.")  # noqa
        result = False
    return result


def validate_player_configs(
    players: list[PlayerConfig],
    n_players: int,
    max_nums_for_roles: dict[ERole, int],
    *,
    alert_func: Callable[[str], None] | None = None
) -> bool | tuple[bool, str]:

    alert_func = alert_func or (lambda _: None)
    result: bool = True

    # assert n_players
    if len(players) > n_players:
        alert_func(f"ValidationError: the number of player configs must be <= {n_players}")  # noqa
        result = False

    # asssert role count
    for role, cnt in Counter([p.role for p in players if p.role]).items():
        thresh = max_nums_for_roles[role]
        print(role, cnt, thresh)
        if cnt > thresh:
            alert_func(f'ValidationError: the number of {role.name} must be <= {thresh}')  # noqa
            result = False

    # names
    for name, cnt in Counter([p.name for p in players]).items():
        if cnt > 1:
            alert_func(f"ValidationError: the name '{name}' is duplicated {cnt} times")  # noqa
            result = False

    # Human count
    if Counter([p.model for p in players])[HUMAN] > 1:
        alert_func(f"ValidationError: the number of '{HUMAN}' in players' models must be 0 or 1.")  # noqa
        result = False

    return result
