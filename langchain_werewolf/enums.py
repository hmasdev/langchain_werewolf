from enum import Enum


class ELanguage(Enum):
    English: str = 'English'
    Chinese: str = 'Chinese'
    Japanese: str = 'Japanese'
    Korean: str = 'Korean'
    French: str = 'French'
    German: str = 'German'
    Spanish: str = 'Spanish'
    Italian: str = 'Italian'
    Dutch: str = 'Dutch'
    Portuguese: str = 'Portuguese'
    Russian: str = 'Russian'


class ESpeakerSelectionMethod(Enum):
    round_robin: str = 'round_robin'
    random: str = 'random'


class ERole(Enum):
    Villager: str = 'Villager'
    Werewolf: str = 'Werewolf'
    Knight: str = 'Knight'
    FortuneTeller: str = 'FortuneTeller'


class ESide(Enum):
    Villager: str = 'Villager'
    Werewolf: str = 'Werewolf'


class EResult(Enum):
    VillagersWin: str = 'Villagers Win'
    WerewolvesWin: str = 'Werewolves Win'


class ESideVictoryCondition(Enum):
    VillagersWinCondition: str = 'All werewolves are excluded from the game'
    WerewolvesWinCondition: str = 'The number of alive werewolves equal or outnumber half of the total number of players'  # noqa


class EChatService(Enum):
    OpenAI: str = 'openai'
    Google: str = 'google'
    Groq: str = 'groq'


class ESystemOutputType(Enum):
    all: str = 'all'
    public: str = 'public'
    off: str = 'off'


class EInputOutputType(Enum):
    standard: str = 'standard'
    click: str = 'click'


class ETimeSpan(Enum):
    day: str = 'day'
    night: str = 'night'
