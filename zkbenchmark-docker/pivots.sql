SELECT
  msm.vector_size,
  MIN(CASE WHEN hw.device = 'vu35p (C1100 board)' THEN msm.runtime_sec END) AS vu35p_runtime,
  MIN(CASE WHEN hw.device = 'vu13p (U250 board)' THEN msm.runtime_sec END) AS vu13p_runtime,
  MIN(CASE WHEN hw.device = 'RTX 3090' THEN msm.runtime_sec END) AS rtx3090_runtime,
  MIN(CASE WHEN hw.device = 'RTX 4090' THEN msm.runtime_sec END) AS rtx4090_runtime
FROM msm_benchmark msm
INNER JOIN hw_platform hw ON msm.runs_on = hw.id
GROUP BY msm.vector_size
ORDER BY msm.vector_size;