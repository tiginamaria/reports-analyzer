"""Service exceptions become HTTP responses here, so routers never build error responses."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.services.errors import ConflictError, InvalidRequestError, NotFoundError, ServiceError

STATUS_CODES: dict[type[ServiceError], int] = {
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ConflictError: status.HTTP_409_CONFLICT,
    InvalidRequestError: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


def register_error_handlers(app: FastAPI) -> None:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        code = next(
            (code for kind, code in STATUS_CODES.items() if isinstance(exc, kind)),
            status.HTTP_400_BAD_REQUEST,
        )
        return JSONResponse({"detail": str(exc)}, status_code=code)

    app.add_exception_handler(ServiceError, handle)
