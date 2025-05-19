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


class EResult(Enum):
    VillagersWin = 'Villagers Win'
    WerewolvesWin = 'Werewolves Win'


class EChatService(Enum):
    OpenAI = 'openai'
    Google = 'google'
    Groq = 'groq'


class ESystemOutputType(Enum):
    all = 'all'
    public = 'public'
    off = 'off'


class EInputOutputType(Enum):
    none = 'none'
    standard = 'standard'
    click = 'click'


class ETimeSpan(Enum):
    day = 'day'
    night = 'night'
