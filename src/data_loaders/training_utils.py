import attr

__all__ = [
    "LooperException",
    "StopLoopingException",
    "EarlyStoppingException",
    "EarlyStopping",
]

class LooperException(Exception):
    """Base class for exceptions encountered during looping"""

    pass

class StopLoopingException(LooperException):
    """Base class for exceptions which should stop training in this module."""

    pass

@attr.s(auto_attribs=True)
class EarlyStoppingException(StopLoopingException):
    """Max Value Exceeded"""

    condition: str

    def __str__(self):
        return f"EarlyStopping: {self.condition}"


@attr.s(auto_attribs=True)
class EarlyStopping:
    """
    Stop looping if a value is stagnant.
    """

    name: str = "EarlyStopping value"
    patience: int = 10

    def __attrs_post_init__(self):
        self.count = 0
        self.value = 0

    def __call__(self, value):
        if value == self.value:
            self.count += 1
            if self.count >= self.patience:
                raise EarlyStoppingException(
                    f"{self.name} has not changed in {self.patience} steps."
                )
        else:
            self.value = value
            self.count = 0