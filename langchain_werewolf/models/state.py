from datetime import datetime
from itertools import chain
from typing import Annotated, Iterable, Literal, TypeVar
from pydantic import BaseModel, Field, field_serializer, field_validator  # noqa
from ..const import RESET
from ..enums import EResult, ETimeSpan
from .general import (
    IdentifiedModel,
    PartialFrozenModel,
    constant_reducer,
    overwrite_reducer,
    reduce_dict,
    reduce_list,
)

T = TypeVar('T')


class MsgModel(BaseModel):
    name: str \
        = Field(..., title="the name of the player")
    timestamp: datetime \
        = Field(title="Timestamp", default_factory=datetime.now)
    message: str \
        = Field(..., title="Message")
    participants: frozenset[str] \
        = Field(title="the names of the participants", default_factory=frozenset)  # noqa
    template: str \
        = Field(
            default='\n'.join([
                "[({timestamp}) {name} spoke to {participants}]",
                '='*30,
                '{message}',
                '',
            ]),
            title="the template of the message",
        )

    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")

    @field_serializer('participants')
    def serialize_participants(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    def format(self) -> str:
        return self.template.format(
            timestamp=self.serialize_timestamp(self.timestamp),
            name=self.name,
            participants=self.serialize_participants(self.participants),
            message=self.message,
        )


class ChatHistoryModel(PartialFrozenModel):
    # FIXME: frozen_fields should be merged with the parent class's frozen_fields  # noqa
    frozen_fields: Annotated[set[str], constant_reducer] = {'frozen_fields', 'names'}  # noqa

    names: frozenset[str]\
        = Field(..., title="the names of the chat participants")
    messages: list[IdentifiedModel[MsgModel]]\
        = Field(title="Chat Messages", default_factory=list)

    @field_serializer('names')
    def serialize_names(self, value: frozenset[str]) -> list[str]:
        return sorted(value)

    @field_validator('messages')
    @classmethod
    def preprocess_messages(
        cls,
        messages: list[IdentifiedModel[MsgModel] | MsgModel],
    ) -> list[IdentifiedModel[MsgModel]]:
        return [
            IdentifiedModel(value=msg)
            if not isinstance(msg, IdentifiedModel) else
            msg
            for msg in messages
        ]


def _reduce_chat_state(
    previous_chat_state: dict[frozenset[str], ChatHistoryModel] | None,
    new_chat_state: dict[frozenset[str], ChatHistoryModel] | None,
) -> dict[frozenset[str], ChatHistoryModel]:
    # initialize
    previous_chat_state = previous_chat_state or {}
    new_chat_state = new_chat_state or {}
    # merge
    for names, chat_history in new_chat_state.items():
        prev_chat_history = previous_chat_state.get(names, ChatHistoryModel(names=names))  # noqa
        prev_chat_history.messages = reduce_list(prev_chat_history.messages, chat_history.messages)  # type: ignore # noqa
        previous_chat_state[names] = prev_chat_history
    return previous_chat_state


def _reduce_votes_current(
    previous_votes: dict[str, str] | None,
    new_votes: dict[str, str] | None,
) -> dict[str, str]:
    previous_votes = previous_votes or {}
    if new_votes is None:
        return previous_votes
    if len(new_votes) == 0:
        # NOTE:
        return {}
    return reduce_dict(previous_votes, new_votes)


class StateModel(PartialFrozenModel):
    # FIXME: frozen_fields should be merged with the parent class's frozen_fields  # noqa
    frozen_fields: Annotated[set[str], constant_reducer] = {'frozen_fields', 'players_names'}  # noqa

    # basic information
    day: Annotated[int, overwrite_reducer]\
        = Field(default=0, title="the day number")
    timespan: Annotated[ETimeSpan, overwrite_reducer]\
        = Field(default=ETimeSpan.day, title="the timespan of the game")
    result: Annotated[EResult | None, overwrite_reducer]\
        = Field(default=None, title="the result of the game")  # noqa

    # chat information
    chat_state: Annotated[
        dict[frozenset[str], ChatHistoryModel],
        _reduce_chat_state,
    ] = Field(title="the chat state", default_factory=dict)  # noqa

    # players information
    alive_players_names: Annotated[list[str], overwrite_reducer]\
        = Field(..., title="the names of the alive players")
    safe_players_names: Annotated[set[str], overwrite_reducer]\
        = Field(title="the names of the safe players", default_factory=set)  # noqa

    # game chat information
    current_speaker: Annotated[str | None, overwrite_reducer]\
        = Field(default=None, title="the name of the current speaker")  # noqa
    n_chat_remaining: Annotated[int | None, overwrite_reducer]\
        = Field(default=None, title="the number of chat remaining")  # noqa

    # vote information
    # TODO: modify the type of daytime_votes_history and nighttime_votes_history: dict to dict[str, str]  # noqa
    daytime_vote_result_history: Annotated[list[IdentifiedModel[str | None]], reduce_list]\
        = Field(title="the history of the daytime vote results", default_factory=list)  # noqa
    daytime_votes_history: Annotated[list[IdentifiedModel[dict]], reduce_list]\
        = Field(title="the votes of each daytime discussion", default_factory=list)  # noqa
    nighttime_vote_result_history: Annotated[list[IdentifiedModel[str | None]], reduce_list]\
        = Field(title="the history of the nighttime vote results", default_factory=list)  # noqa
    nighttime_votes_history: Annotated[list[IdentifiedModel[dict]], reduce_list]\
        = Field(title="the votes of each nighttime discussion", default_factory=list)  # noqa
    daytime_votes_current: Annotated[dict[str, str], _reduce_votes_current]\
        = Field(default_factory=dict, title="the current daytime votes")
    nighttime_votes_current: Annotated[dict[str, str], _reduce_votes_current]\
        = Field(default_factory=dict, title="the current nighttime votes")

    @field_serializer('chat_state')
    def serialize_chat_state(
        self,
        value: dict[frozenset[str], ChatHistoryModel],
    ) -> dict[str, dict]:
        return {
            '|'.join(sorted(key)): value.model_dump()
            for key, value in value.items()
        }

    @field_serializer('safe_players_names')
    def serialize_safe_players_names(self, value: set[str]) -> list[str]:
        return sorted(value)

    @field_serializer('result')
    def serialize_result(self, value: EResult | None) -> str:
        if value is None:
            return 'None'
        return value.value

    @field_validator('daytime_votes_history')
    @classmethod
    def preprocess_votes_history(
        cls,
        votes_history: list[IdentifiedModel[dict[str, str]] | dict[str, str]],  # noqa
    ) -> list[IdentifiedModel[dict[str, str]]]:
        return [
            IdentifiedModel(value=votes)
            if not isinstance(votes, IdentifiedModel) else
            votes
            for votes in votes_history
        ]

    @field_validator('nighttime_votes_history')
    @classmethod
    def preprocess_nighttime_votes_history(
        cls,
        votes_history: list[IdentifiedModel[dict[str, str]] | dict[str, str]],  # noqa
    ) -> list[IdentifiedModel[dict[str, str]]]:
        return [
            IdentifiedModel(value=votes)
            if not isinstance(votes, IdentifiedModel) else
            votes
            for votes in votes_history
        ]

    def validate_state(self, raise_exception: bool = True) -> bool:
        try:
            assert len(self.nighttime_vote_result_history) == len(self.nighttime_votes_history), f'assert len(self.nighttime_vote_result_history) == len(self.nighttime_votes_history) failed: {len(self.nighttime_vote_result_history)} != {len(self.nighttime_votes_history)}, nighttime_vote_result_history: {self.nighttime_vote_result_history}, nighttime_votes_history: {self.nighttime_votes_history}'  # noqa
            assert len(self.daytime_vote_result_history) == len(self.daytime_votes_history), f'assert len(self.daytime_vote_result_history) == len(self.daytime_votes_history) failed: {len(self.daytime_vote_result_history)} != {len(self.daytime_votes_history)}, daytime_vote_result_history: {self.daytime_vote_result_history}, daytime_votes_history: {self.daytime_votes_history}'  # noqa
        except AssertionError:
            if raise_exception:
                raise ValueError('validation failed')
            return False
        return True


def create_dict_to_reset_state(*args, **kwargs) -> dict[str, object]:
    return {
        'safe_players_names': set(),
        'current_speaker': None,
        'n_chat_remaining': None,
        'daytime_votes_current': {},
        'nighttime_votes_current': {},
    }


def create_dict_to_record_chat(
    sender: str,
    participants: Iterable[str],
    message: str,
) -> dict[str, dict[frozenset[str], ChatHistoryModel]]:
    """Generate a dictionary to update the chat state attribute of StateModel

    Args:
        sender (str): message sender
        participants (Iterable[str]): sender's and receivers' names
        message (str): message content

    Returns:
        dict[str, dict[frozenset[str], ChatHistoryModel]]: dictionary to update the chat state attribute of StateModel
    """  # noqa
    participants = frozenset(chain([sender], participants))
    return {
        'chat_state': {
            participants: ChatHistoryModel(
                names=participants,
                messages=[IdentifiedModel[MsgModel](value=MsgModel(
                    name=sender,
                    message=message,
                    participants=participants
                ))],
            ),
        }
    }


def create_dict_to_update_day(
    day: int,
) -> dict[str, int]:
    return {'day': day}


def create_dict_to_update_timespan(
    timespan: ETimeSpan,
) -> dict[str, ETimeSpan]:
    return {'timespan': timespan}


def create_dict_to_add_safe_player(
    name: str,
) -> dict[str, set[str]]:
    return {
        'safe_players_names': {name},
    }


def create_dict_to_update_current_speaker(
    name: str | None,
) -> dict[str, str | None]:
    return {
        'current_speaker': name,
    }


def create_dict_to_update_chat_remaining_number(
    n_chat_remaining: int,
) -> dict[str, int]:
    return {
        'n_chat_remaining': n_chat_remaining,
    }


def create_dict_to_update_daytime_vote_result_history(
    vote_result: str | None,
) -> dict[str, list[IdentifiedModel[str | None]]]:
    return {
        'daytime_vote_result_history': [IdentifiedModel[str | None](value=vote_result)],  # noqa
    }


def create_dict_to_update_nighttime_vote_result_history(
    vote_result: str | None,
) -> dict[str, list[IdentifiedModel[str | None]]]:
    return {
        'nighttime_vote_result_history': [IdentifiedModel[str | None](value=vote_result)],  # noqa
    }


def create_dict_to_update_daytime_votes_current(
    votes: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {
        'daytime_votes_current': votes,
    }


def create_dict_to_update_nighttime_votes_current(
    votes: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {
        'nighttime_votes_current': votes,
    }


def create_dict_to_update_daytime_votes_history(
    votes: dict[str, str | Literal[RESET]],  # type: ignore  # noqa
) -> dict[str, list[dict[str, str | Literal[RESET]]]]:  # type: ignore  # noqa
    return {'daytime_votes_history': [votes]}


def create_dict_to_update_nighttime_votes_history(
    votes: dict[str, str | Literal[RESET]],  # type: ignore  # noqa
) -> dict[str, list[dict[str, str | Literal[RESET]]]]:  # type: ignore  # noqa
    return {'nighttime_votes_history': [votes]}


def create_dict_to_update_alive_players(
    names: Iterable[str],
) -> dict[str, list[str]]:
    return {
        'alive_players_names': list(names),
    }


def create_dict_to_update_result(
    result: EResult | None,
) -> dict[str, EResult | None]:
    return {
        'result': result,
    }


def create_dict_without_state_updated(
    state: StateModel,
) -> dict[str, object]:
    return {'chat_state': {}}


def get_related_chat_histories(
    name: str,
    state: StateModel,
) -> dict[frozenset[str], ChatHistoryModel]:
    return {
        chat_history.names: chat_history
        for chat_history in state.chat_state.values()
        if name in chat_history.names
    }


def _integrate_chat_histories(
    *chat_history: ChatHistoryModel,
) -> list[IdentifiedModel[MsgModel]]:
    return sorted([
        IdentifiedModel[MsgModel](
            id=message.id,
            value=MsgModel(
                name=message.value.name,
                timestamp=message.value.timestamp,
                message=message.value.message,
                participants=history.names,
            ))
        for history in chat_history
        for message in history.messages
    ], key=lambda x: x.value.timestamp)


def _get_specific_chat(
    names: frozenset[str],
    state: StateModel,
) -> ChatHistoryModel:
    return state.chat_state.get(names, ChatHistoryModel(names=names))


def get_related_messsages_with_id(
    name: str | Iterable[str],
    state: StateModel,
) -> list[IdentifiedModel[MsgModel]]:
    if isinstance(name, str):
        return _integrate_chat_histories(*get_related_chat_histories(name, state).values())  # noqa
    return _integrate_chat_histories(_get_specific_chat(frozenset(name), state))  # noqa


def get_related_messsages(
    name: str | Iterable[str],
    state: StateModel,
) -> list[MsgModel]:
    return [msg.value for msg in get_related_messsages_with_id(name, state)]
