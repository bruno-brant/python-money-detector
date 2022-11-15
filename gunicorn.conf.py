# Configure the gunicorn server. See the gunicorn documentation for
# more information.
# http://docs.gunicorn.org/en/stable/settings.html

# Network settings
bind = '0.0.0.0:8000'
workers = 4

# Logging configuration
log_level = "debug"
errorlog = "-"
accesslog = "-"
