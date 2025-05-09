from . import __package__ as package_name
from .enums import EChatService, ELanguage

# library
PACKAGE_NAME: str = package_name

# general
RESET: str = "_____RESET_____"
GAME_MASTER_NAME: str = 'GameMaster'
BASE_LANGUAGE: ELanguage = ELanguage.English

# CLI
CLI_PROMPT_SUFFIX: str = '>>> '
CLI_PROMPT_COLOR: str = 'black'
CLI_ECHO_COLORS: tuple[str, ...] = (
    'white',
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'bright_black',
    'bright_red',
    'bright_green',
    'bright_yellow',
    'bright_blue',
    'bright_magenta',
    'bright_cyan',
    'bright_white',
)

# Player
DEFAULT_PLAYER_PREFIX: str = 'Player'
CUSTOM_PLAYER_PREFIX: str = 'CustomPlayer'

# llm
DEFAULT_MODEL: str = 'gpt-4o-mini'
MODEL_SERVICE_MAP: dict[str, EChatService] = {
    'gpt-3.5-turbo': EChatService.OpenAI,
    'gpt-4': EChatService.OpenAI,
    'gpt-4-turbo': EChatService.OpenAI,
    'gpt-4o': EChatService.OpenAI,
    'gpt-4o-mini': EChatService.OpenAI,
    'gemini-1.5-flash': EChatService.Google,
    "gemini-pro-vision": EChatService.Google,
    'gemini-pro': EChatService.Google,
    'gemma2-9b-it': EChatService.Groq,
    'gemma2-7b-it': EChatService.Groq,
    'llama3-groq-70b-8192-tool-use-preview': EChatService.Groq,
    'llama3-groq-8b-8192-tool-use-preview': EChatService.Groq,
    'llama-3.1-70b-versatile': EChatService.Groq,
    'llama-3.1-8b-instant': EChatService.Groq,
    'llama-guard-3-8b': EChatService.Groq,
    'llava-v1.5-7b-4096-preview': EChatService.Groq,
    'llama3-70b-8192': EChatService.Groq,
    'llama3-8b-8192': EChatService.Groq,
    'mixtral-8x7b-32768': EChatService.Groq,
}
VALID_MODELS: tuple[str, ...] = tuple(MODEL_SERVICE_MAP.keys())
SERVICE_APIKEY_ENVVAR_MAP: dict[EChatService, str] = {
    EChatService.Google: 'GOOGLE_API_KEY',
    EChatService.Groq: 'GROQ_API_KEY',
    EChatService.OpenAI: 'OPENAI_API_KEY',
}
