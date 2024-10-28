import streamlit as st
from streamlit_langchain_werewolf.session import Session


def quit_view(session: Session) -> None:
    if session.game_session_state is not None:
        if session.game_session_state.game_process is not None:
            session.game_session_state.game_process.kill()
        session.game_session_state.messages_queue_from_game_to_player.close()
    st.session_state.clear()
    st.rerun()
