"""
Output - nomina uno stream come output del modello.
"""

from nnodely.core.stream import Stream


class Output(Stream):
    """Expose ``stream`` as an output of the model, under ``name``.

    Output names are the keys of the model's results and the names used to
    refer to outputs elsewhere, for example in feedback mappings.
    """

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
