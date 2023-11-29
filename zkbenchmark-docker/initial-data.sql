
DO $$
DECLARE my_time TIMESTAMP;
DECLARE my_git_id VARCHAR(100);
DECLARE my_frequency_MHz INT;
DECLARE my_comment VARCHAR(1024);
DECLARE ff_id INT;
DECLARE hw_id INT;

BEGIN
    SELECT NOW() AT TIME ZONE 'Israel' INTO my_time;
    my_git_id := 'unknown';
    my_frequency_MHz := 250;
    my_comment := 'pre-loaded points, varying scalars';
    SELECT finite_field.id INTO ff_id from finite_field where name = 'BLS12-377';
    SELECT hw_platform.id INTO hw_id from hw_platform where device = 'vu35p (C1100 board)';
    
    RAISE NOTICE 'HW Id %', hw_id;
    RAISE NOTICE 'Field Id %', ff_id;
    RAISE NOTICE 'Test time %', my_time;

INSERT INTO msm_benchmark (test_timestamp, git_id, frequency_MHz, vector_size, coefficient_C, batch_size, runtime_sec, power_Watt, chip_temp_C, comment, runs_on, uses)
VALUES 
  (my_time, my_git_id, my_frequency_MHz, 1024, 12, 1000, 923.61E-6, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 2048, 12, 1000, 1.03E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 4096, 12, 1000, 1.24E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 8192, 12, 1000, 1.66E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 16384, 12, 1000, 2.52E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 32768, 12, 1000, 4.39E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 65536, 12, 1000, 7.99E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 131072, 12, 1000, 15.18E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 262144, 12, 1000, 29.48E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 524288, 12, 1000, 58.13E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 1048576, 12, 1000, 140.88E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 2097152, 12, 100, 271.63E-3, 0.0, 0.0, my_comment, hw_id, ff_id),
  (my_time, my_git_id, my_frequency_MHz, 4194304, 12, 100, 527.24E-3, 0.0, 0.0, my_comment, hw_id, ff_id);

END$$;






