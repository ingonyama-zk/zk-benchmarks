#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "${INGO_BENCHMARKS_DB_HOST}"
echo "${INGO_BENCHMARKS_DB_PORT}"
echo "${INGO_BENCHMARKS_DB_NAME}"
echo "${INGO_BENCHMARKS_DB_USER}"
#DB_PASS="${INGO_BENCHMARKS_DB_PASSWORD}"


read -p "Do you want to proceed? (Yes/No) " answer
if [[ $answer =~ ^[Nn] ]]; then
    echo "Terminating."
    exit 1
fi

export PGPASSWORD=$INGO_BENCHMARKS_DB_PASSWORD
psql -U $INGO_BENCHMARKS_DB_USER -d $INGO_BENCHMARKS_DB_NAME -h $INGO_BENCHMARKS_DB_HOST -p $INGO_BENCHMARKS_DB_PORT
