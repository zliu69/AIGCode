__all__ = [
    "AIGCodeError",
    "AIGCodeConfigurationError",
    "AIGCodeCliError",
    "AIGCodeEnvironmentError",
    "AIGCodeNetworkError",
    "AIGCodeCheckpointError",
]


class AIGCodeError(Exception):
    """
    Base class for all custom AIGCode exceptions.
    """


class AIGCodeConfigurationError(AIGCodeError):
    """
    An error with a configuration file.
    """


class AIGCodeCliError(AIGCodeError):
    """
    An error from incorrect CLI usage.
    """


class AIGCodeEnvironmentError(AIGCodeError):
    """
    An error from incorrect environment variables.
    """


class AIGCodeNetworkError(AIGCodeError):
    """
    An error with a network request.
    """


class AIGCodeCheckpointError(AIGCodeError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class AIGCodeThreadError(Exception):
    """
    Raised when a thread fails.
    """
