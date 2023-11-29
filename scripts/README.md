# Benchmark database tables

### NTT 

```toml
project=<project_name> # Name of the project or GitHub repository
git_id=<commit_hash> # Specific commit hash of the project
frequency_MHz=<operating_frequency_in_megahertz> # Frequency at which the hardware was operating during the benchmark
vector_size=<size_of_vector_processed> # Size of the vector processed in the benchmark
batch_size=<number_of_operations_in_batch> # Number of operations processed in a single batch
runtime_sec=<execution_time_in_seconds> # Total time taken for the benchmark in seconds
power_Watt=<power_consumption_in_watts> # Power consumption during the benchmark in Watts
chip_temp_C=<chip_temperature_in_celsius> # Temperature of the chip during the benchmark in Celsius
comment=<additional_comments> # Any additional comments or notes
runs_on=<hardware_used> # Description of the hardware used for the benchmark
uses=<curve_or_algorithm_used> # The cryptographic curve or algorithm used in the benchmark
```

### MSM

```toml
project=<project_name> # Name of the project or GitHub repository
git_id=<commit_hash> # Specific commit hash of the project
frequency_MHz=<operating_frequency_in_megahertz> # Frequency at which the hardware was operating during the benchmark
vector_size=<size_of_vector_processed> # Size of the vector processed in the benchmark
coefficient_C=<coefficient_value> # Coefficient value used in the benchmark
batch_size=<number_of_operations_in_batch> # Number of operations processed in a single batch
runtime_sec=<execution_time_in_seconds> # Total time taken for the benchmark in seconds
power_Watt=<power_consumption_in_watts> # Power consumption during the benchmark in Watts
chip_temp_C=<chip_temperature_in_celsius> # Temperature of the chip during the benchmark in Celsius
comment=<additional_comments> # Any additional comments or notes
runs_on=<hardware_used> # Description of the hardware used for the benchmark
uses=<curve_or_algorithm_used> # The cryptographic curve or algorithm used in the benchmark
```

### Hash algorithm's (poseidon)

```toml
project=<project_name> # Name of the project or GitHub repository
git_id=<commit_hash> # Specific commit hash of the project
frequency_MHz=<operating_frequency_in_megahertz> # Frequency at which the hardware was operating during the benchmark
tree_height=<height_of_merkle_tree> # Height of the Merkle tree used in the benchmark
batch_size=<number_of_operations_in_batch> # Number of operations processed in a single batch
runtime_sec=<execution_time_in_seconds> # Total time taken for the benchmark in seconds
power_Watt=<power_consumption_in_watts> # Power consumption during the benchmark in Watts
chip_temp_C=<chip_temperature_in_celsius> # Temperature of the chip during the benchmark in Celsius
comment=<additional_comments> # Any additional comments or notes
runs_on=<hardware_used> # Description of the hardware used for the benchmark
uses=<curve_or_algorithm_used> # The cryptographic curve or algorithm used in the benchmark
```
