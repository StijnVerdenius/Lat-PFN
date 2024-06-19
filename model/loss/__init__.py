
from .simple import CrossEntropyLoss

losses: dict[str, callable] = {
    "cross_entropy": CrossEntropyLoss,
}
