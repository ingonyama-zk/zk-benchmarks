#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

#QUERY="SELECT * FROM ntt_benchmark;"
QUERY="SELECT 
    team,  
    project,
    test_timestamp,
    git_id,
    frequency_MHz,
    vector_size,
    batch_size,
    runtime_sec,
    power_Watt,
    chip_temp_C,
    comment,
    hw_platform.device AS device,
    finite_field.name AS finite_field
FROM 
    ntt_benchmark
JOIN 
    hw_platform ON ntt_benchmark.runs_on = hw_platform.id
JOIN 
    finite_field ON ntt_benchmark.uses = finite_field.id;
"

################################
# Do not modify below this line
################################

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