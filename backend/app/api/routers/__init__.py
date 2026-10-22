"""One router per resource. Handlers are async and thin: parse the request, call a service,
shape the response; background work is scheduled here because it is an HTTP concern."""

from app.api.routers import chat, datapoints, reports, system

__all__ = ["chat", "datapoints", "reports", "system"]
