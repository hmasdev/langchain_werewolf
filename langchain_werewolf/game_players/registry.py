from langchain_core.runnables import Runnable

from .base import (
    BaseGamePlayerRole,
    BasePlayerSideMixin,
    GamePlayerRunnableInputModel,
)


class PlayerSideRegistry:

    _registry: dict[str, type[BasePlayerSideMixin]] = {}

    @classmethod
    def register(
        cls,
        side_class: type[BasePlayerSideMixin],
    ) -> type[BasePlayerSideMixin]:
        """Register a new side class
        Args:
            side_class (type[BasePlayerSideMixin]):
                a subclass of BasePlayerSideMixin representing a player's side.

        Returns:
            type[BasePlayerSideMixin]: side_class itself

        Raises:
            TypeError: if side_class is not a subclass of BasePlayerSideMixin
        """
        if not issubclass(side_class, BasePlayerSideMixin):
            raise TypeError(f"The class {side_class} is not a subclass of {BasePlayerSideMixin}")  # noqa
        key = side_class.side
        cls._registry[key] = side_class
        return side_class

    @classmethod
    def unregister(cls, side_class: type[BasePlayerSideMixin]) -> None:
        """Unregister a side class
        Args:
            side_class (type[BasePlayerSideMixin]):
                a subclass of BasePlayerSideMixin representing a player's side.
        """
        registry_clone: dict[str, type[BasePlayerSideMixin]] = cls._registry.copy()  # noqa
        for key, cls_side in registry_clone.items():
            if cls_side is side_class:
                del cls._registry[key]

    @classmethod
    def get_keys(cls) -> list[str]:
        """Get all registered keys

        Returns:
            list[str]: all registered keys
        """
        return list(cls._registry.keys())

    @classmethod
    def get_class(cls, key: str) -> type[BasePlayerSideMixin]:
        """Get player's side class by key

        Args:
            key (str): the key identifying a registered player side class
        Returns:
            type[BasePlayerSideMixin]: the side class associated with the given key
        Raises:
            KeyError: if the key is not registered
        """  # noqa
        try:
            return cls._registry[key]
        except KeyError:
            raise KeyError(f"The side {key} is not registered.")

    @classmethod
    def get_key(cls, side_class: type[BasePlayerSideMixin]) -> str:
        """Get the key by side class

        Args:
            side_class (type[BasePlayerSideMixin]): the class of the side
        Returns:
            str: the key associated with the side class
        Raises:
            KeyError: the side class is not registered
        """
        for key, cls_side in cls._registry.items():
            if cls_side is side_class:
                return key
        raise ValueError(f"The side {side_class} is not registered.")

    @classmethod
    def initialize(cls) -> type["PlayerSideRegistry"]:
        """
        Initialize the registry with all sides in the following module registered:
            `langchain_werewolf.game_players.player_sides`
        """  # noqa
        from .player_sides import __all__  # noqa
        return cls


class PlayerRoleRegistry:

    _registry: dict[str, type[BaseGamePlayerRole]] = {}

    @classmethod
    def register(
        cls,
        role_class: type[BaseGamePlayerRole],
    ) -> type[BaseGamePlayerRole]:
        """Register a new role class
        Args:
            role_class (type[BasePlayerSideMixin]): the class of the role
        """
        if not issubclass(role_class, BaseGamePlayerRole):
            raise TypeError(f"The class {role_class} is not a subclass of {BaseGamePlayerRole}")  # noqa
        if not issubclass(role_class, BasePlayerSideMixin):
            raise TypeError(f"The class {role_class} is not a subclass of {BasePlayerSideMixin}")  # noqa
        key = role_class.role
        cls._registry[key] = role_class
        return role_class

    @classmethod
    def unregister(cls, role_class: type[BaseGamePlayerRole]) -> None:
        """Unregister a role class
        Args:
            role_class (type[BasePlayerSideMixin]): the class of the role
        """
        registry_clone: dict[str, type[BaseGamePlayerRole]] = cls._registry.copy()  # noqa
        for key, cls_role in registry_clone.items():
            if cls_role is role_class:
                del cls._registry[key]

    @classmethod
    def get_keys(cls) -> list[str]:
        """Get all registered role names
        Returns:
            list[str]: all registered keys
        """
        return list(cls._registry.keys())

    @classmethod
    def get_class(cls, key: str) -> type[BaseGamePlayerRole]:
        """Get the role class by key
        Args:
            key (str): the name of the role
        Returns:
            type[BaseGamePlayerRole]: the class of the key
        Raises:
            KeyError: the key is not registered
        """
        try:
            return cls._registry[key]
        except KeyError:
            raise KeyError(f"The role {key} is not registered.")

    @classmethod
    def get_key(cls, role_class: type[BaseGamePlayerRole]) -> str:
        """Get the key by role class
        Args:
            role_class (type[BaseGamePlayerRole]): the class of the role
        Returns:
            str: the key associated with the role class
        Raises:
            KeyError: the role class is not registered
        """
        for key, cls_role in cls._registry.items():
            if cls_role is role_class:
                return key
        raise ValueError(f"The role {role_class} is not registered.")

    @classmethod
    def create_player(
        cls,
        key: str,
        name: str,
        runnable: Runnable[GamePlayerRunnableInputModel, str],
        **kwargs,
    ) -> BaseGamePlayerRole:
        """Create a player by key
        Args:
            key (str): the name of the role
            name (str): the name of the player
            runnable (Runnable[GamePlayerRunnableInputModel, str]): the runnable of the player
            **kwargs: other arguments to pass to the role
        Returns:
            BaseGamePlayer: the player
        Raises:
            KeyError: the key is not registered
        """  # noqa
        try:
            role_cls = cls.get_class(key)
            return role_cls(name=name, runnable=runnable, **kwargs)  # noqa
        except KeyError:
            raise KeyError(f"The role {key} is not registered.")

    @classmethod
    def initialize(cls) -> type["PlayerRoleRegistry"]:
        """
        Initialize the registry with all roles in the following module registered:
            `langchain_werewolf.game_players.player_roles`
        """  # noqa
        from .player_roles import __all__  # noqa
        return cls
