from __future__ import annotations
from multiprocessing import Queue, Process
from typing import Literal, overload
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, SkipValidation
from langchain_werewolf.models.state import StateModel
from streamlit_langchain_werewolf.model import GameSetupModel, StremlitMessageModel  # noqa


class _GameSessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[StremlitMessageModel] = Field(default_factory=list, title='Received and sesnt messages')  # noqa

    result_queue: SkipValidation[Queue] = Field(title='Result Queue', description='Queue for the result', default_factory=Queue)  # noqa
    messages_queue_from_game_to_player: SkipValidation[Queue] = Field(title='Messages Queue', description='Queue for messages from the game', default_factory=Queue)  # noqa
    messages_queue_from_player_to_game: SkipValidation[Queue] = Field(title='Messages Queue', description='Queue for messages from the player', default_factory=Queue)  # noqa

    game_process: Process | None = Field(default=None, title='Game process', description='Process for the game')  # noqa


class Session:

    game_setting: GameSetupModel | None
    game_session_state: _GameSessionState | None
    result: StateModel | None

    @overload
    @staticmethod
    def get_session(raise_exception: Literal[True] = True) -> Session:
        ...

    @overload
    @staticmethod
    def get_session(raise_exception: Literal[False] = False) -> Session | None:
        ...

    @staticmethod
    def get_session(raise_exception: bool = True) -> Session | None:
        state: Session = st.session_state.get('state', None)
        if raise_exception and state is None:
            raise ValueError('State is None')
        return state

    def __new__(cls, *args, **kwargs):
        instance = cls.get_session(raise_exception=False)
        instance = instance or super().__new__(cls, *args, **kwargs)
        return instance

    def __init__(self):
        if self.get_session(raise_exception=False) is None:
            st.session_state.state = self
        self.game_setting = None
        self.game_session_state = None
        self.game_threads = None
        self.result = None

    @classmethod
    def game_started(cls) -> bool:
        session: Session | None = cls.get_session(raise_exception=False)
        return (
            session is not None
            and session.game_setting is not None
            and session.game_session_state is not None
            and session.game_session_state.game_process is not None
        )

    @classmethod
    def game_finished(cls) -> bool:
        session: Session | None = cls.get_session(raise_exception=False)
        if session is None:
            return False

        if session.result is not None:
            return True
        if any([
            session.game_session_state is None,
            session.game_session_state is not None
            and session.game_session_state.result_queue.empty(),
        ]):
            return False

        session.result = session.game_session_state.result_queue.get()  # type: ignore # noqa
        if session is not None and session.result is None:
            raise ValueError('Invalid result')
        return True
