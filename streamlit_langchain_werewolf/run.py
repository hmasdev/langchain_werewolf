from functools import partial
import json
from logging import getLogger, Logger
from multiprocessing import Queue, Process
from langchain_werewolf.main import main as langchain_werewolf_main
from langchain_werewolf.models.config import PlayerConfig
from langchain_werewolf.models.state import MsgModel
from streamlit_langchain_werewolf.const import HUMAN
from streamlit_langchain_werewolf.io import echo, prompt
from streamlit_langchain_werewolf.model import GameSetupModel
from streamlit_langchain_werewolf.session import Session, _GameSessionState


def _wrapped_langchain_werewolf_main(
    *args,
    _result_queue: Queue,
    **kwargs,
) -> None:
    state = langchain_werewolf_main(*args, **kwargs)
    _result_queue.put(state)


def _formatter(msg: MsgModel) -> str:
    return json.dumps(msg.model_dump(include={'name', 'timestamp', 'participants', 'message'}))  # noqa


def _initialize_game_session(
    *,
    logger: Logger = getLogger(__name__),
) -> _GameSessionState:
    q1 = Queue()  # type: ignore
    q2 = Queue()  # type: ignore
    game_session_state = _GameSessionState(
        messages_queue_from_game_to_player=q1,
        messages_queue_from_player_to_game=q2,
    )
    logger.info('Game session state has been initialized')
    return game_session_state


def _attach_queue_to_player(
    session: Session,
) -> None:

    cfg: GameSetupModel | None = session.game_setting
    state = session.game_session_state
    assert cfg
    assert state

    prompt_ = partial(
        prompt,
        send_queue=state.messages_queue_from_game_to_player,  # noqa
        receive_queue=state.messages_queue_from_player_to_game,  # noqa
        session=session,
    )
    echo_ = partial(
        echo,
        send_queue=state.messages_queue_from_game_to_player,  # noqa
        session=session,
    )

    if any([p.model == HUMAN for p in cfg.config.players]):
        human_idx = [p.model for p in cfg.config.players].index(HUMAN)
        cfg.config.players[human_idx] = PlayerConfig(
            **(
                cfg.config.players[human_idx].model_dump()
                | dict(
                    player_output_interface=echo_,
                    player_input_interface=prompt_,
                    formatter=_formatter,
                )
            )
        )
        cfg.system_input_interface = prompt_
    else:
        cfg.system_input_interface = prompt_
        cfg.system_output_interface = echo_


def _start_game(
    session: Session,
    logger: Logger = getLogger(__name__),
) -> Process:

    cfg: GameSetupModel | None = session.game_setting
    state = session.game_session_state
    assert cfg
    assert state

    game_process = Process(
        target=_wrapped_langchain_werewolf_main,
        kwargs=dict(
            n_players=cfg.n_players,
            n_werewolves=cfg.n_werewolves,
            n_knights=cfg.n_knights,
            n_fortune_tellers=cfg.n_fortune_tellers,
            config=cfg.config,
            seed=cfg.seed,
            model=cfg.model,
            system_language=cfg.system_language,
            system_output_level=cfg.system_output_level,
            system_input_interface=cfg.system_input_interface,
            system_output_interface=cfg.system_output_interface,
            system_formatter=_formatter,
            system_font_color=None,
            player_font_colors=None,
            _result_queue=state.result_queue,
        ),
    )
    game_process.daemon = True
    game_process.start()
    logger.info('Game process has been started')
    return game_process


def run(
    session: Session,
    logger: Logger = getLogger(__name__),
) -> None:
    session.game_session_state = _initialize_game_session()
    _attach_queue_to_player(session)
    session.game_session_state.game_process = _start_game(session, logger)
    logger.info('Game has been started')
