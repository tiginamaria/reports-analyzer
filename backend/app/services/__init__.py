"""Use cases. Services own the unit of work (commit, rollback) and raise the exceptions in
`services.errors`; they know nothing about HTTP. Database access is async; blocking work
(hashing and moving an upload, rendering a page with PyMuPDF) runs in worker threads."""
