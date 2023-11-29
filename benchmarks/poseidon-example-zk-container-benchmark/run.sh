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

# Run the benchmark and capture its output in the json_data variable
json_data=$(/icicle-benchmark/build/benchmark --benchmark_time_unit=s --benchmark_format=json)

# Extract context 
project=$(echo "$json_data" | jq -r '.context.project')

runs_on=$(echo "$json_data" | jq -r '.context.runs_on')
frequency_MHz=$(echo "$json_data" | jq -r '.context.frequency_MHz')
uses=$(echo "$json_data" | jq -r '.context.uses')
comment=$(echo "$json_data" | jq -r '.context.comment')



echo "Project: $project"
echo "GitID: $git_id"
echo "Frequency_MHz: $frequency_MHz"
echo "Comment: $comment"
echo "Runs on: $runs_on"
echo "Uses: $uses"

test_timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Timestamp: $test_timestamp"


nof_benchmarks=$(echo "$json_data" | jq '.benchmarks | length')
echo "nof_benchmarks $nof_benchmarks"

for ((nof_benchmark = 0; nof_benchmark < nof_benchmarks; nof_benchmark++)); do
    echo "Benchmark $nof_benchmark"
  
    benchmark=$(echo "$json_data" | jq ".benchmarks[$nof_benchmark]")
    # echo $benchmark

    benchmark_name=$(echo "$benchmark" | jq '.name')
    echo "Complete benchmark name: $benchmark_name"

    # Arguments: the complete benchmark name ends with "/"-separated arguments

    tree_height=$(echo "$benchmark_name" | cut -d'/' -f2)
    echo "Tree Height: $tree_height"
    
    runtime_sec=$(echo "$benchmark" | jq '.real_time')
    echo "Run time (sec): $runtime_sec"
    
    batch_size=$(echo "$benchmark" | jq '.iterations')
    echo "Batch size: $batch_size"

    power_Watt=$(echo "$benchmark" | jq '.PowerUsage')
    echo "Power (Watt): $power_Watt"

    chip_temp_C=$(echo "$benchmark" | jq '.Temperature')
    echo "Chip temperature (C): $chip_temp_C"


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

done

QUERY="SELECT * FROM poseidon_benchmark ORDER BY id DESC LIMIT 10;"

PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$QUERY"


exit




