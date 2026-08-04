"""
Output - nomina uno stream come output del modello.
"""

from nnodely.core.stream import Stream


class Output(Stream):
    def __init__(self, name: str, stream: Stream) -> None:
        super().__init__(
            name=name,
            dim=stream.shape.dim,
            time=stream.shape.time,
            seq=stream.shape.seq,
            preds=[stream],
        )
