export PGPASSWORD=$INGO_BENCHMARKS_DB_PASSWORD
psql -U $INGO_BENCHMARKS_DB_USER -d $INGO_BENCHMARKS_DB_NAME -h $INGO_BENCHMARKS_DB_HOST -p $INGO_BENCHMARKS_DB_PORT