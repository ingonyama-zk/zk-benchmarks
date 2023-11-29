\connect ingo_benchmarks

CREATE TABLE grostl_benchmark (
  id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project VARCHAR(1024),
  test_timestamp TIMESTAMP,
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

GRANT SELECT ON ALL TABLES IN SCHEMA public TO guest_user;
