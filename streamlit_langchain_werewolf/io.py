from logging import getLogger, Logger
from multiprocessing import Queue
from uuid import uuid4
from streamlit_langchain_werewolf.session import Session
from streamlit_langchain_werewolf.model import StremlitMessageModel


def echo(
    msg: str,
    send_queue: Queue,  # Queue[StremlitMessageModel]
    session: Session | None = None,
    logger: Logger = getLogger(__name__),
) -> None:
    session = session or Session.get_session()
    message = StremlitMessageModel(token='', message=msg, response_required=False)  # noqa
    send_queue.put(message)
    logger.info(f'Message has been sent: {message}')


def prompt(
    msg: str,
    send_queue: Queue,  # Queue[StremlitMessageModel]
    receive_queue: Queue,  # Queue[StremlitMessageModel]
    session: Session | None = None,
    logger: Logger = getLogger(__name__),
) -> str:
    session = session or Session.get_session()
    token = str(uuid4())

    # send message
    smsg = StremlitMessageModel(token=token, message=msg, response_required=True)  # noqa
    send_queue.put(smsg)

    # receive message
    rmsg: StremlitMessageModel | None = None
    for _ in range(5):
        rmsg = receive_queue.get(timeout=120)
        if (
            isinstance(rmsg, StremlitMessageModel)
            and rmsg.message
            and rmsg.token == token
        ):
            break
    else:
        logger.error(f'Timeout: {smsg}')
    logger.info(f'Message has been received: {rmsg}')
    return rmsg.message if rmsg else ''
