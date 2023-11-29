# Ingonyama Benchmark tool kit

We maintain an internal SQL database to store the benchmarks we run on our products. Currently we benchmark our solutions for  MSM, Poseidon, NTT, and Modulo Multiplications. 

## Who can use the benchmarking tool kit?

The benchmarking tool kit makes it easy to setup and self host a benchmark database. We also offer tool to integrate into your CI/CD and criterion benchmarks recoding of the benchmarks and logging them to the benchmark database.

Developers can use the benchmarking tool kit to fail CI/CD workflows if minimum benchmark requirements are not met, collect historical benchmarking data, compare performance of their system to others and compare libraries one to another.

Researchers can generate status reports and retrieve performance metrics of different implementations on workloads.


## What hardware do we include?

We support primarily GPUs and FPGAs. Some example of GPUs we commonly benchmark are `RTX 3090`, `RTX 4090` and we also have benchmarked many of the Xilinx FPGAs `vu35p (C1100 board)` and `vu13p (U250 board)`.


## How to use the database?

If you wish to self host you can refer to these [instructions](./zkbenchmark-docker/README.md).

The rest of this document describes how you can use Ingonyama's hosted benchmark database.

### Install PostgreSQL client

PostgreSQL also offers [GUI based clients](https://www.pgadmin.org/) if you dont want to use the command line.



#### Ubuntu
```bash
sudo apt-get install postgresql-client
```

#### MacOS
You will need [`Homebrew`](https://brew.sh/) package manager on your Mac.  
If not, install it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the recommendations of the installation script to add `brew` to your path. 


Next install the database client 

```bash
brew install libpq
brew link --force libpq
```

### Define Database Connection environment variables

We maintain two databases: test and production. Play with the test database before commiting to production one.

Create a `.env` file based of of [`.env.example`](/.env.example)

If you want to connect to Ingonyama's database use the following values to gain access to a readonly user:

```sh
INGO_BENCHMARKS_DB_HOST=benchmarks.ingonyama.com
INGO_BENCHMARKS_DB_PORT=5432
INGO_BENCHMARKS_DB_NAME=ingo_benchmarks
INGO_BENCHMARKS_DB_USER=guest_user
```

Export the environment variables call: `source .env`

If you wish to connect with `psql` use the following command:

```sh
psql -h $INGO_BENCHMARKS_DB_HOST -p $INGO_BENCHMARKS_DB_PORT -U $INGO_BENCHMARKS_DB_USER -d $INGO_BENCHMARKS_DB_NAME -c "SELECT * FROM poseidon_benchmark;"
```

## Upload benchmarks

### Using shell scripts
Shell scripts will take the benchmark data from user-specified test files. 

1. Copy examples from `./scripts/example-*-data.sh` and modify with your own data.  
For example, for `msm`

```bash
cat ./scripts/example-msm-data.sh
project='ALEO'
git_id='7918b1c'
frequency_MHz=250
vector_size=8192
coefficient_C=12
batch_size=1
runtime_sec='6.25e-5'
power_Watt='450.0'
chip_temp_C='0.0'
comment='ALEO'
runs_on='RTX 4090'
uses='BLS12-377'% 
```

2. Run provided script `./scripts/add-*.sh` with your benchmark data as an argument. 

For example, for `msm`

```bash
./scripts/add-msm.sh ./scripts/example-msm-data.sh
```

### Using Rust's Criterion

The interface changes fast. At this moment, the best way is talk to me (Stas Polonsky) directly.

## Generating reports

### Summarize all benchmarks in Excel spreadsheet

We use `Python3` to produce summaries. You might need to install few dependencies: 

```bash
pip3 install psycopg2-binary
pip3 install openpyxl
```

### Pivot tables

Pivot tables compare performance (e.g. runtime, power) for different input data size and hardware.
For example, to report on MSM, run `./scripts/pivot-msm.sh `:


```
 vector_size | vu35p_runtime | vu13p_runtime | rtx3090_runtime | rtx4090_runtime 
-------------+---------------+---------------+-----------------+-----------------
        1024 |    0.00092361 |               |                 |                
        2048 |       0.00103 |               |                 |                
        4096 |       0.00124 |               |                 |                
        8192 |         0.001 |        0.0005 |        0.000143 |        6.25e-05
       16384 |       0.00252 |               |                 |                
       32768 |       0.00439 |               |                 |                
       65536 |       0.00799 |               |                 |                
      131072 |       0.01518 |               |                 |                
      262144 |       0.02948 |               |                 |                
      524288 |       0.05813 |               |                 |                
     1048576 |       0.14088 |               |                 |                
     2097152 |       0.27163 |               |                 |                
     4194304 |       0.52724 |               |                 |                
(13 rows)
```

### Data dumps

To list all data for a solution, run `./scripts/view-*.sh`. For example for MSM run `./scripts/view-msm.sh`

## Database design details: ER diagrams

https://docs.google.com/presentation/d/1Fxs34TuPmix6cejpqKrX1NYrxf2ExYK7LYH6OkGUkSU/edit?usp=share_link
