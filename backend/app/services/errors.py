"""Exceptions services raise; `api.errors` maps them to HTTP responses."""


class ServiceError(Exception):
    """Base class; the message is safe to show to the user."""


class NotFoundError(ServiceError):
    pass


class ConflictError(ServiceError):
    """The operation does not apply to the entity in its current state."""


class InvalidRequestError(ServiceError):
    """The request is well-formed but refers to something that does not exist."""


class UploadRejected(ServiceError):
    """A file in a batch was not accepted; the message is the reason shown to the user."""
