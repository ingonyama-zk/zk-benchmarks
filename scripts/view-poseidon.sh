#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

QUERY="SELECT * FROM poseidon_benchmark;"

# Set database connection parameters
if  [ -z "${INGO_BENCHMARKS_DB_HOST}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_PORT}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_NAME}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_USER}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_PASSWORD}" ]; then
    echo "Define INGO_BENCHMARKS environment variables"
    exit 1
fi

DB_HOST="${INGO_BENCHMARKS_DB_HOST}"
DB_PORT="${INGO_BENCHMARKS_DB_PORT}"
DB_NAME="${INGO_BENCHMARKS_DB_NAME}"
DB_USER="${INGO_BENCHMARKS_DB_USER}"
DB_PASS="${INGO_BENCHMARKS_DB_PASSWORD}"

PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$QUERY"
