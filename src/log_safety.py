"""Keep exception details that may quote resume text, prompts or provider payloads out of the logs.

A provider error message can contain the request or the model's partial output, and a traceback prints that message.
The filter below leaves the log line itself and appends only the exception CLASS name; it drops the traceback and
the exception message. It is attached to the root logger and to the parser/recommender loggers, so calls such as
``logging.exception("Resume summarization failed")`` stay useful (what failed, which exception type) without leaking.
"""
import logging

_LOGGERS = ("", "src.parser", "src.recommender", "backend.main")


class ExceptionClassOnly(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            exc_type = record.exc_info[0]
            name = getattr(exc_type, "__name__", "Exception")
            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)
            record.msg = "%s (%s)" % (message, name)
            record.args = None
            record.exc_info = None
            record.exc_text = None
        record.stack_info = None
        return True


def install_exception_redaction():
    """Idempotent: attach one ExceptionClassOnly filter to each relevant logger."""
    for name in _LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(f, ExceptionClassOnly) for f in target.filters):
            target.addFilter(ExceptionClassOnly())
