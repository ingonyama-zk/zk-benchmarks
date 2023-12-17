#!/bin/bash

# Exit immediately on error
set -e

DB_HOST=${INGO_BENCHMARKS_DB_HOST}
DB_PORT=${INGO_BENCHMARKS_DB_PORT}
DB_NAME=${INGO_BENCHMARKS_DB_NAME}
DB_USER=${INGO_BENCHMARKS_DB_USER}
DB_PASS=${INGO_BENCHMARKS_DB_PASS}


# testing this Icicle version
git_id=$(cd /icicle && git rev-parse --short HEAD)
echo "Icicle GitID: $git_id"

echo "Running the benchmarks and capturing the output in the file benchmark.json"
#/icicle-benchmark/build/benchmark --benchmark_time_unit=s  --benchmark_out_format=json --benchmark_out=benchmark.json

json_data=$(<benchmark.json)

echo "Extracting context..."

team=$(jq -r '.context.team' benchmark.json)
echo "Team: $team"

project=$(jq -r '.context.project' benchmark.json)
echo "Project: $project"

runs_on=$(jq -r '.context.runs_on' benchmark.json)
echo "Runs on: $runs_on"

frequency_MHz=$(jq -r '.context.frequency_MHz' benchmark.json)
echo "Frequency_MHz: $frequency_MHz"

uses=$(jq -r '.context.uses' benchmark.json)
echo "Uses: $uses"

#vector_size=$(jq -r '.context.vector_size' benchmark.json)
#echo "Vector Size: $vector_size"

comment=$(jq -r '.context.comment' benchmark.json)
echo "Comment: $comment"

test_timestamp=$(jq -r '.context.date' benchmark.json)
#test_timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Timestamp: $test_timestamp"

echo "Extracting benchmarks..."

nof_benchmarks=$(echo "$json_data" | jq '.benchmarks | length')
echo "Number of benchmarks: $nof_benchmarks"

rm --force benchmark.sql
for ((nof_benchmark = 0; nof_benchmark < nof_benchmarks; nof_benchmark++)); do
  echo "Benchmark $nof_benchmark"
  
  benchmark=$(jq ".benchmarks[$nof_benchmark]" benchmark.json)
  # echo $benchmark

  benchmark_name=$(echo "$benchmark" | jq '.name')
  echo "Complete benchmark name: $benchmark_name"

  # Arguments: the complete benchmark name ends with "/"-separated arguments

  #vector_size=$(echo "$benchmark_name" | cut -d'/' -f2)
  vector_size=$(echo "$benchmark_name" | awk -F'[/""]' '{print 2^$3}')
  echo "Vector size: $vector_size"
    
  runtime_sec=$(echo "$benchmark" | jq '.real_time')
  echo "Run time (sec): $runtime_sec"
    
  batch_size=$(echo "$benchmark" | jq '.iterations')
  echo "Batch size: $batch_size"

  power_Watt=$(echo "$benchmark" | jq '.PowerUsage')
  echo "Power (Watt): $power_Watt"

  chip_temp_C=$(echo "$benchmark" | jq '.Temperature')
  echo "Chip temperature (C): $chip_temp_C"

  QUERY="INSERT INTO ntt_benchmark (
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
  runs_on,
  uses
)
VALUES (
  '$team',
  '$project',
  '$test_timestamp',
  '$git_id',
  $frequency_MHz,
  $vector_size,
  $batch_size,
  $runtime_sec,
  $power_Watt,
  $chip_temp_C,
  '$comment',
  (SELECT id FROM hw_platform WHERE device = '$runs_on'),
  (SELECT id FROM finite_field WHERE name = '$uses')
);
"
  echo "$QUERY" >> benchmark.sql

  # Execute the psql command to insert the row into the ntt_benchmark table
  # PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c $QUERY

done


#QUERY="SELECT * FROM add_benchmark ORDER BY id DESC LIMIT 10;"
#PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$QUERY"

exit




