from enum import Enum


class ELanguage(Enum):
    English = 'English'
    Chinese = 'Chinese'
    Japanese = 'Japanese'
    Korean = 'Korean'
    French = 'French'
    German = 'German'
    Spanish = 'Spanish'
    Italian = 'Italian'
    Dutch = 'Dutch'
    Portuguese = 'Portuguese'
    Russian = 'Russian'


class ESpeakerSelectionMethod(Enum):
    round_robin = 'round_robin'
    random = 'random'


class ERole(Enum):
    Villager = 'Villager'
    Werewolf = 'Werewolf'
    Knight = 'Knight'
    FortuneTeller = 'FortuneTeller'


class ESide(Enum):
    Villager = 'Villager'
    Werewolf = 'Werewolf'


class EResult(Enum):
    VillagersWin = 'Villagers Win'
    WerewolvesWin = 'Werewolves Win'


class ESideVictoryCondition(Enum):
    VillagersWinCondition = 'All werewolves are excluded from the game'
    WerewolvesWinCondition = 'The number of alive werewolves equal or outnumber half of the total number of players'  # noqa


class EChatService(Enum):
    OpenAI = 'openai'
    Google = 'google'
    Groq = 'groq'


class ESystemOutputType(Enum):
    all = 'all'
    public = 'public'
    off = 'off'


class EInputOutputType(Enum):
    standard = 'standard'
    click = 'click'


class ETimeSpan(Enum):
    day = 'day'
    night = 'night'
