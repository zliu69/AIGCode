__all__ = [
    "AIGCcodeError",
    "AIGCcodeConfigurationError",
    "AIGCcodeCliError",
    "AIGCcodeEnvironmentError",
    "AIGCcodeNetworkError",
    "AIGCcodeCheckpointError",
]


class AIGCcodeError(Exception):
    """
    Base class for all custom AIGCcode exceptions.
    """


class AIGCcodeConfigurationError(AIGCcodeError):
    """
    An error with a configuration file.
    """


class AIGCcodeCliError(AIGCcodeError):
    """
    An error from incorrect CLI usage.
    """


class AIGCcodeEnvironmentError(AIGCcodeError):
    """
    An error from incorrect environment variables.
    """


class AIGCcodeNetworkError(AIGCcodeError):
    """
    An error with a network request.
    """


class AIGCcodeCheckpointError(AIGCcodeError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class AIGCcodeThreadError(Exception):
    """
    Raised when a thread fails.
    """
