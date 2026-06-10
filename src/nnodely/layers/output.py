"""
Output - nomina uno stream come output del modello.
"""

from nnodely.core.stream import Stream
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Output(Stream):
    def __init__(self, name: str, stream: Stream) -> None:
        super().__init__(
            name=name,
            seq=stream.seq,
            time=stream.time,
            dim=stream.dim,
            preds=[stream],
        )
        self._closed_loop = {}
