"""Data access: every SQL query of the request path lives here, one repository per
aggregate. Repositories wrap an `AsyncSession` and never commit; the service that owns the
unit of work does."""
