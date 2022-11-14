# Configure the gunicorn server. See the gunicorn documentation for
# more information.
# http://docs.gunicorn.org/en/stable/settings.html

# Network settings
bind = '0.0.0.0:443'
workers = 4

# Logging configuration
log_level = "debug"
errorlog = "-"
accesslog = "-"


# SSL/TLS configuration
certfile = '/app/server.crt'
keyfile = '/app/server.key'
ssl_version = 'TLS_SERVER'
do_handshake_on_connect = True
