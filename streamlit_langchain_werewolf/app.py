from logging import getLogger, Logger
import time
import streamlit as st
from streamlit_langchain_werewolf.logic import messasges_require_response
from streamlit_langchain_werewolf.model import StremlitMessageModel
from streamlit_langchain_werewolf.run import run
from streamlit_langchain_werewolf.session import Session
from streamlit_langchain_werewolf.view import (
    finish_view,
    header_view,
    input_chat_message_view,
    messages_view,
    setup_view,
    quit_view,
)


def _main(session: Session, logger: Logger = getLogger(__name__)) -> None:

    # initial run
    if not session.game_started():
        run(session)
    # validate
    assert session.game_session_state is not None

    # init
    queue_from_game2player = session.game_session_state.messages_queue_from_game_to_player  # noqa
    queue_from_player2game = session.game_session_state.messages_queue_from_player_to_game  # noqa

    # show past messages
    msg: StremlitMessageModel | None = None
    for msg in session.game_session_state.messages:
        messages_view(msg)
    if msg and messasges_require_response(msg):
        if message := input_chat_message_view():
            sent_msg = StremlitMessageModel(
                token=msg.token,
                message=message,
                response_required=False,
            )
            session.game_session_state.messages.append(sent_msg)
            messages_view(sent_msg)
            queue_from_player2game.put(sent_msg)

    # show messages
    while not session.game_finished():
        time.sleep(0.1)

        if session.game_finished():
            finish_view(session)
            break

        # get message from queue
        if queue_from_game2player.empty():
            continue
        msg = queue_from_game2player.get()  # type: ignore
        # validate message
        if not isinstance(msg, StremlitMessageModel):
            logger.error(f'Invalid message: {msg}')
            continue
        # show
        session.game_session_state.messages.append(msg)
        messages_view(msg)
        # check if input required
        if messasges_require_response(msg):
            st.rerun()


def _quit(session: Session, logger: Logger = getLogger(__name__)) -> None:
    quit_view(session)
    logger.info('Session has been terminated')


def main(logger: Logger = getLogger(__name__)) -> None:

    # initialize
    session: Session
    try:
        session = Session.get_session()
    except ValueError:
        logger.warning('Session is not found. Creating a new session...')
        session = Session()

    # header
    header_view()

    # setup
    session.game_setting = setup_view(session, disabled=session.game_started())

    # buttons run/quit
    cols = st.columns(3)
    start = cols[0].button('Start Game', disabled=session.game_started())
    cols[1].button('Reset', on_click=_quit, args=(session,))  # noqa
    cols[2].button('Reload', on_click=st.rerun)

    # main
    if start or session.game_started():  # noqa
        _main(session)
