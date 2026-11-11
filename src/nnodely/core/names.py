"""Picking the objects of a model by name, by index or by themselves.

Names are how the user reaches a Parameter, a model or a stream, and nothing
keeps two objects from sharing one: layers named alike by hand, a block and
the model calling it, two copies of one loaded model. So every call taking a
name looks it up among the objects the graph reaches, in that order, and
refuses a name several carry; the lists the user reads them from are also
indexed by position, as the index column of their tables shows. A key is
resolved to its object where the user gives it - by :func:`resolve` - and
inside nnodely the objects themselves are used, never their names.

This module imports nothing of nnodely, so every module can use it.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

#: ``(object, index, where)``: an object a key may pick, its index in the
#: list it is picked from - or None - and where it is, for the listing of a
#: name several carry.
Candidate = tuple[Any, "int | None", str]


class NameNotFound(KeyError, ValueError):
    """A key that picks nothing: a ``KeyError``, as a missing key of a table
    is, and a ``ValueError``, as a wrong argument of a call is."""

    def __str__(self) -> str:
        # KeyError quotes its message; this one reads as a sentence.
        return str(self.args[0]) if self.args else ""


def by_name_or_index(key, owner: str) -> bool:
    """Whether ``key`` picks by name: a string; an int or a slice picks by
    position, as in a list. Anything else - a name and an index together -
    raises ``TypeError``."""
    if isinstance(key, str):
        return True
    if isinstance(key, (int, slice)) and not isinstance(key, bool):
        return False
    raise TypeError(
        f"{owner}: an item is picked by its name, a string, or by its index, "
        f"an int - the index column of the table - not both; got {key!r}."
    )


def pick(
    candidates: Sequence[Candidate],
    name: str,
    *,
    kind: str,
    owner: str,
    hint: str,
    kinds: str | None = None,
) -> list:
    """The object of ``candidates`` that carries the name ``name``, as a
    list: ``[]`` for none. When several carry it, raises ``ValueError``
    listing them, and ``hint``: how to pick one, or to rename one. ``kinds``
    is the plural of ``kind``, its name with an s by default."""
    found = [
        (item, index, where) for item, index, where in candidates if item.name == name
    ]
    if len(found) <= 1:
        return [item for item, _, _ in found]
    indices = [index for _, index, _ in found]
    at = "" if None in indices else f", at indices {indices}"
    listing = "; ".join(
        f"{'' if index is None else f'[{index}] '}{name} {where}".rstrip()
        for _, index, where in found
    )
    raise ValueError(
        f"{owner}: the name {name!r} is ambiguous, {len(found)} "
        f"{kinds or kind + 's'} carry it{at}: {listing}. {hint}"
    )


def resolve(
    key,
    candidates: Sequence[Candidate],
    *,
    kind: str,
    owner: str,
    hint: str,
    kinds: str | None = None,
    indices: bool = False,
    objects: type | tuple[type, ...] | None = None,
    same: Callable[[Any, Any], bool] | None = None,
    missing: Callable[[Any], str] | None = None,
) -> Any:
    """The object of ``candidates`` the user's ``key`` picks - the one place
    a key becomes an object:

    - a name, a string: the one candidate carrying it; a name several carry
      raises ``ValueError`` listing them (:func:`pick`);
    - with ``indices``, an int or a slice: by position, as in a list;
    - with ``objects``, an instance of those types: the candidate it is - or,
      with ``same(candidate, key)``, the one it stands for, as an Input
      stands for the copies a build makes of it.

    Any other key raises ``TypeError``. A key that picks nothing raises
    :class:`NameNotFound`, with ``missing(key)`` as its message when given."""
    items = [item for item, _, _ in candidates]
    plural = kinds or kind + "s"
    if objects is not None and isinstance(key, objects):
        match = same or (lambda item, given: item is given)
        found = next((item for item in items if match(item, key)), None)
        if found is None:
            raise NameNotFound(
                missing(key)
                if missing
                else f"{owner}: {getattr(key, 'name', key)!r} is not one of its "
                f"{plural}."
            )
        return found
    if indices and not by_name_or_index(key, owner):
        return items[key]
    if not isinstance(key, str):
        article = "an" if kind[:1].lower() in "aeiou" else "a"
        raise TypeError(
            f"{owner}: {article} {kind} is picked by its name, a string"
            f"{', or by itself' if objects is not None else ''}; got {key!r}."
        )
    picked = pick(candidates, key, kind=kind, kinds=kinds, owner=owner, hint=hint)
    if not picked:
        raise NameNotFound(
            missing(key)
            if missing
            else f"{owner}: no {kind} named {key!r}; the names are "
            f"{[item.name for item in items]}."
        )
    return picked[0]


class Table(list):
    """A list of the objects of a model a table prints - its Parameters,
    Inputs, outputs, models or minimizers - picked by name or by index, the
    index column of the table, as :func:`resolve` picks them. A subclass
    sets ``_kind`` and ``_attr``, the name of its list on a model, and gives
    where each item is with :meth:`_where`."""

    _kind = "item"
    _attr = "items"
    _title = ""

    def _owner(self) -> str:
        return self._title or self._attr.capitalize()

    def _where(self, index: int, item) -> str:
        """Where ``item``, at ``index``, is: for the listing of a name several
        carry."""
        return ""

    def _candidates(self) -> list[Candidate]:
        return [
            (item, index, self._where(index, item)) for index, item in enumerate(self)
        ]

    def __getitem__(self, key):  # type: ignore[override]
        """An item by its name, or by its index - the index column of the
        table - as in a list. A name several items carry raises
        ``ValueError`` listing them, with their index."""
        return resolve(
            key,
            self._candidates(),
            kind=self._kind,
            owner=self._owner(),
            hint=f"Pick one by its index, {self._attr}[i], or give it a name of "
            f"its own, {self._attr}[i].name = 'new_name'.",
            indices=True,
        )
