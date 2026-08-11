"""
Output - nomina uno stream come output del modello.
"""

from nnodely.core.stream import Stream
from nnodely.core.registry import register_node


@register_node
class Output(Stream):
    def __init__(self, name: str, stream: Stream) -> None:
        super().__init__(
            name=name,
            dim=stream.shape.dim,
            time=stream.shape.time,
            seq=stream.shape.seq,
            preds=[stream],
        )

    def get_config(self) -> dict:
        return {
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict, preds=None) -> "Output":
        if preds is None or len(preds) != 1:
            raise ValueError(
                f"Output '{config['name']}' requires exactly one predecessor."
            )

        return cls(
            name=config["name"],
            stream=preds[0],
        )
