# Temporary stage for conditional copying
FROM alpine as config-stage
WORKDIR /config
# Try to copy the PostgreSQL config files
# Use a wildcard to avoid failure if files do not exist
COPY postgresql.conf* pg_hba.conf* /config/

# Main stage: Use the official PostgreSQL image as the base
FROM postgres:latest

# Copy configuration files from the temporary stage
# Only the files that exist will be copied
# COPY --from=config-stage /config/ /var/lib/postgresql/data/

# Set environment variables (optional)
# ENV POSTGRES_DB=mydb
# ENV POSTGRES_USER=myuser
# ENV POSTGRES_PASSWORD=mypassword

# copy custom configurations
COPY postgresql.conf /etc/postgresql/postgresql.conf
COPY pg_hba.conf /etc/postgresql/pg_hba.conf

# Copy the initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Expose the PostgreSQL default port
EXPOSE 5432

# Set a custom working directory (optional)
WORKDIR /app

# Install additional packages (optional)
RUN apt-get update && apt-get install -y vim

# Define a startup command (optional)
#CMD ["postgres"]
