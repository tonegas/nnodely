"""Type aliases for nnodely package."""

from typing import TypeAlias

from nnodely.core.stream import Stream

# Type alias for stream operands (used in Stream arithmetic operations)
StreamOperand: TypeAlias = Stream | list[int | float | list] | int | float
