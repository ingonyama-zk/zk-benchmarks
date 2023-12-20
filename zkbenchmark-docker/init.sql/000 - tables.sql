CREATE DATABASE ingo_benchmarks;

\connect ingo_benchmarks

CREATE TYPE hw_category AS ENUM ('CPU', 'GPU', 'FPGA');

CREATE TABLE hw_platform (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  category hw_category,
  device VARCHAR(100)
);

INSERT INTO hw_platform (category, device)
VALUES 
  ('FPGA', 'vu35p (C1100 board)'),
  ('FPGA', 'vu13p (U250 board)'),
  ('GPU', 'NVIDIA GeForce RTX 3090'),
  ('GPU', 'NVIDIA GeForce RTX 4090')
;

CREATE TABLE finite_field (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR(100)
);

INSERT INTO finite_field (name)
VALUES 
  ('BLS12-377'),
  ('BN254'),
  ('BLS12-381')
;


CREATE TYPE multiply_type AS ENUM ('scalar', 'point_field');

CREATE TABLE multiply_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  multiply multiply_type,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  vector_size BIGINT,
  operation_factor BIGINT,
  batch_size INT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  comment VARCHAR(1024),
  -- computed: operations per second
  ops REAL GENERATED ALWAYS AS ( operation_factor * vector_size / runtime_sec ) STORED,
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);

CREATE TABLE add_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  vector_size BIGINT,
  operation_factor BIGINT,
  batch_size INT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  comment VARCHAR(1024),
  -- computed: operations per second
  ops REAL GENERATED ALWAYS AS ( operation_factor * vector_size / runtime_sec ) STORED,
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);

CREATE TABLE msm_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  vector_size BIGINT,
  coefficient_C INT,
  batch_size INT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  comment VARCHAR(1024),
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);

CREATE TABLE ntt_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  vector_size BIGINT,
  batch_size INT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  comment VARCHAR(1024),
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);


CREATE TABLE poseidon_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  tree_height BIGINT,
  batch_size INT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  comment VARCHAR(1024),
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);

CREATE TABLE grostl_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  team VARCHAR(1024),
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
  git_repository VARCHAR(1024),
  git_id VARCHAR(100),
  frequency_MHz INT,
  tree_height BIGINT,
  batch_size BIGINT,
  runtime_sec REAL,
  power_Watt REAL,
  chip_temp_C REAL,
  throughput_GiB_sec REAL,
  msg_size BIGINT,
  comment VARCHAR(1024),
  -- relationships
  runs_on INT,
  FOREIGN KEY (runs_on) REFERENCES hw_platform (id),
  uses INT,
  FOREIGN KEY (uses) REFERENCES finite_field (id)
);




