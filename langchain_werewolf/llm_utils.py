from enum import Enum
from functools import lru_cache
from logging import Logger, getLogger
from operator import attrgetter
from langchain.output_parsers import (
    EnumOutputParser,
    RetryWithErrorOutputParser,
)

from langchain.output_parsers.retry import NAIVE_RETRY_WITH_ERROR_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .const import DEFAULT_MODEL, MODEL_SERVICE_MAP, BASE_LANGUAGE
from .enums import EChatService, ELanguage


_service2cls: dict[EChatService, type[BaseChatModel]] = {
    EChatService.OpenAI: ChatOpenAI,
    EChatService.Google: ChatGoogleGenerativeAI,
    EChatService.Groq: ChatGroq,
}


@lru_cache(maxsize=None)
def create_chat_model(
    llm: BaseChatModel | str | None = DEFAULT_MODEL,
    seed: int | None = None,
    logger: Logger = getLogger(__name__),
    **kwargs,
) -> BaseChatModel:
    """Create a ChatModel instance.

    Args:
        llm (BaseChatModel | str | None, optional): ChatModel instance or model name. Defaults to DEFAULT_MODEL.
        seed (int, optional): Random seed. Defaults to None.
        logger (Logger, optional): Logger. Defaults to getLogger(__name__).

    Raises:
        ValueError: Unknown model name

    Returns:
        BaseChatModel: ChatModel instance

    Note:
        seed is used only when llm is a str.
        The same parameters return the same instance.
    """  # noqa
    llm = llm or DEFAULT_MODEL
    if isinstance(llm, str):
        try:
            if seed is not None:
                return _service2cls[MODEL_SERVICE_MAP[llm]](model=llm, seed=seed, **kwargs)  # type: ignore  # noqa
            else:
                return _service2cls[MODEL_SERVICE_MAP[llm]](model=llm, **kwargs)  # type: ignore  # noqa
        except TypeError:
            logger.warning(f'{llm} does not support seed.')
            return _service2cls[MODEL_SERVICE_MAP[llm]](model=llm, **kwargs)  # type: ignore  # noqa
        except KeyError:
            raise ValueError(f'Unknown model name: {llm}')
    else:
        return llm


def extract_name(
    message: str,
    valid_names: list[str],
    context: str | None = None,
    chat_model: BaseChatModel | Runnable[str, str] | str | None = None,
    seed: int | None = None,
    max_retry: int = 5,
) -> str:
    """Extract a valid name from the message.

    Args:
        message (str): The message to extract the name from.
        valid_names (list[str]): The list of valid names.
        context (str | None, optional): The context like conditions and restrictions. Defaults to None.
        chat_model (BaseChatModel | Runnable[str, str] | str | None, optional): The chat model. Defaults to None.
        seed (int | None, optional): The random seed. Defaults to None.
        max_retry (int, optional): The maximum number of retries. Defaults to 5.

    Returns:
        str: _description_
    """  # noqa
    if chat_model is None:
        chat_model = ChatOpenAI(model='gpt-4o-mini')
    if isinstance(chat_model, str):
        chat_model = create_chat_model(chat_model, seed=seed)

    base_llm_chain: Runnable[str, str]
    if chat_model.OutputType == str:
        base_llm_chain = chat_model  # type: ignore
    else:
        base_llm_chain = chat_model | RunnableLambda(attrgetter('content'))

    chain = RetryWithErrorOutputParser.from_llm(
        parser=EnumOutputParser(enum=Enum(
            'ValidNames',
            dict(zip(valid_names, valid_names)) | {'Nobody': 'None'},  # noqa
        )),  # type: ignore
        llm=RunnableLambda(attrgetter('text')) | base_llm_chain,  # type: ignore # noqa
        prompt=NAIVE_RETRY_WITH_ERROR_PROMPT,
        max_retries=max_retry,
    )

    prompt = '\n'.join([
        'You are the best at consolidating opinions and drawing conclusions.',  # noqa
        f'Your task is to extract a valid name, where valid names are {valid_names}.',  # noqa
        context or '',
        'Extract the valid name from the following message:',
        '```text',
        message,
        '```',
        'Extract the valid name from the above message.',
    ])

    return chain.parse_with_prompt(  # type: ignore
        completion=prompt,
        prompt_value=StringPromptValue(text=message),
    ).value  # type: ignore


def create_translator_runnable(
    to_language: ELanguage,
    chat_llm: BaseChatModel | Runnable[str, str],
    prompt_template: PromptTemplate | str = '''
You are the best translator in the world.
Translate the following text into {language}.
----------
{text}
----------
Translated the above text into {language}.
Output only the translated text.
''',
) -> Runnable[str, str]:
    f"""Create a translator runnable.

    Args:
        to_language (ELanguage): The target language.
        chat_llm (BaseChatModel | Runnable[str, str]): The chat model or llm-like object.
        prompt_template (PromptTemplate | str, optional): prompt template for translation. Defaults to ''' You are the best translator in the world. Translate the following text into {{language}}. ---------- {{text}} ---------- Translated the above text into {{language}}. Output only the translated text. '''.

    Returns:
        Runnable[str, str]: The translator runnable.

    Note:
        when to_language is {BASE_LANGUAGE}, the chain is a passthrough chain.
    """  # noqa
    if to_language == BASE_LANGUAGE:
        return RunnablePassthrough().with_types(input_type=str, output_type=str)  # type: ignore # noqa
    if isinstance(prompt_template, str):
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['text', 'language'],
        )
    # create chain and translator
    chain: Runnable[str, str] = RunnableParallel(
        text=RunnablePassthrough(),
        language=RunnableLambda(lambda _: to_language),
    ) | prompt | chat_llm | RunnableBranch(
        (
            lambda x: hasattr(x, 'content'),
            RunnableLambda(attrgetter('content')),
        ),
        RunnablePassthrough(),
    )
    return chain.with_types(input_type=str, output_type=str)
