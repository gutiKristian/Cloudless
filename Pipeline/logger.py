import logging

log = logging.getLogger(__name__)


# where the logs should go
file_handler = logging.FileHandler("sentinel_pipeline_logs.log")
console_handler = logging.StreamHandler()

log.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

fmt_console = '%(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

console_formatter = logging.Formatter(fmt_console)
file_formatter = logging.Formatter(fmt_file)

console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)
