from typing import Annotated, Any, Generic, TypeVar
import uuid
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar('T')


def overwrite_reducer(old: T, new: T) -> T:
    return new


def constant_reducer(old: T, new: T) -> T:
    return old


def _generate_unique_string():
    """Generate a unique string

    Returns:
        str: a unique string
    """
    return str(uuid.uuid4())


class PartialFrozenModel(BaseModel):
    # FIXME: frozen_fields should be a class variable
    frozen_fields: Annotated[set[str], constant_reducer] = {'frozen_fields'}  # noqa
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.frozen_fields and name in self.__dict__:
            raise TypeError(f"{name} is a frozen field. You cannot change it.")  # noqa
        return super().__setattr__(name, value)


class IdentifiedModel(PartialFrozenModel, Generic[T]):
    # FIXME: frozen_fields should be merged with the parent class's frozen_fields  # noqa
    frozen_fields: Annotated[set[str], constant_reducer] = {'frozen_fields', 'id'}  # noqa

    id: str = Field(title="object id", default_factory=_generate_unique_string)  # noqa
    value: T = Field(title="the value of the model")


def reduce_dict(
    old: dict[str, T] | None,  # type: ignore
    new: dict[str, T] | None,  # type: ignore
) -> dict[str, T]:
    old = old or {}
    new = new or {}
    old.update(new)
    return old


def reduce_list(
    old: list[T | IdentifiedModel[T]] | None,  # type: ignore
    new: list[T | IdentifiedModel[T]] | None,  # type: ignore
) -> list[IdentifiedModel[T]]:
    # ref. https://langchain-ai.github.io/langgraph/how-tos/subgraph/
    # initialize
    old = old or []
    new = new or []
    old_: list[IdentifiedModel[T]] = []
    new_: list[IdentifiedModel[T]] = []

    # convert to IdentifiedModel
    for orglst, newlst in [(old, old_), (new, new_)]:
        for val in orglst:
            if not isinstance(val, IdentifiedModel):
                val = IdentifiedModel[type(val)](value=val)  # type: ignore
            newlst.append(val)  # type: ignore

    # merge
    old_idx_by_id = {val.id: i for i, val in enumerate(old_)}
    merged = old_.copy()
    for val in new_:
        if (existing_idx := old_idx_by_id.get(val.id)) is not None:
            merged[existing_idx] = val
        else:
            merged.append(val)
    return merged
