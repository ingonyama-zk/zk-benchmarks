#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 data-file-name"
    exit 1
fi

# Source the data-file
source "$1"

echo "Project: $project"
echo "GitID: $git_id"
echo "Tree Height: $tree_height"
echo "Batch size: $batch_size"
echo "Run time (sec): $runtime_sec"
echo "Power (Watt): $power_Watt"
echo "Chip temperature (C): $chip_temp_C"
echo "Comment: $comment"
echo "Runs on: $runs_on"
echo "Uses: $uses"

test_timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Timestamp: $test_timestamp"



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

echo "---"
echo "Database: $DB_NAME"
echo "---"

read -p "Do you want to proceed? (Yes/No) " answer
if [[ $answer =~ ^[Nn] ]]; then
    echo "Terminating."
    exit 1
fi

# Execute the psql command to insert the row into the poseidon_benchmark table
PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
INSERT INTO poseidon_benchmark (project, test_timestamp, git_id, frequency_MHz, tree_height, batch_size, runtime_sec, power_Watt, chip_temp_C, comment, runs_on, uses)
VALUES (
  '$project',
  '$test_timestamp',
  '$git_id',
  $frequency_MHz,
  $tree_height,
  $batch_size,
  $runtime_sec,
  $power_Watt,
  $chip_temp_C,
  '$comment',
  (SELECT id FROM hw_platform WHERE device = '$runs_on'),
  (SELECT id FROM finite_field WHERE name = '$uses')
);
"
