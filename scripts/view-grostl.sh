#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

QUERY="SELECT id, project, batch_size, msg_size, runtime_sec, throughput_gib_sec, test_timestamp, git_id, frequency_mhz, power_watt, chip_temp_c, comment  FROM grostl_benchmark;"

# Set database connection parameters
if  [ -z "${INGO_BENCHMARKS_DB_HOST}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_PORT}" ] || \
    [ -z "${INGO_BENCHMARKS_DB_NAME}" ]; then
    echo "Define INGO_BENCHMARKS environment variables"
    exit 1
fi

DB_HOST="${INGO_BENCHMARKS_DB_HOST}"
DB_PORT="${INGO_BENCHMARKS_DB_PORT}"
DB_NAME="${INGO_BENCHMARKS_DB_NAME}"
DB_USER="guest_user"

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$QUERY"
