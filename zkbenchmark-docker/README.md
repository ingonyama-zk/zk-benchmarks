# Self hosting ZK Benchmark Database

## Introduction

This docker configuration can be used to deploy your own zk-benchmarking database.

## Prerequisites

- Docker and Docker Compose installed on your machine.
- Basic understanding of Docker and PostgreSQL.

## Setup Steps

### 1. Dockerfile Configuration
The Dockerfile sets up a PostgreSQL environment. It is based on the official PostgreSQL image, copies an initialization script (`init.sql`), and installs additional packages like `vim`.

### 2. Docker Compose Configuration
`docker-compose.yml` is used to define and run the service. It builds the container, sets up the environment variable for the database password, maps ports, and configures data persistence.

### 3. Building and Running the Container
First, ensure that the `INGO_BENCHMARKS_DB_PASSWORD` environment variable is set in your shell or `.env` file. 

The Docker setup now includes a multi-stage build process that conditionally copies custom PostgreSQL configuration files into the image. This allows for using predefined settings without manually editing configuration files within the running container.


Review and modify if necessary configuration files `postgresql.conf` and `pg_hba.conf`.

Run the following command in the directory containing your `docker-compose.yml`:

```sh
docker-compose up -d
```

This command builds the image, creates the volume `zk_benchmark_data`, and starts the container.

To stop the container, use 

```sh
docker-compose down
```


### 4. Configuring PostgreSQL

Connect to the running container:

```sh
docker exec -it zk-benchmark-db bash
```

`exit` from the container and restart it:

```sh
docker-compose restart zk-benchmark-db
```

### 5. Database Modification (Optional)

To apply migrations or run SQL scripts, you can connect to the PostgreSQL database using:

```sh
docker exec -it zk-benchmark-db psql -U postgres -d ingo_benchmarks
```

You can then execute SQL commands or use `\i` to run SQL scripts.

### 6. Remote Query Testing

If you want to connect to your database from a remote machine run the following script.

```sh
psql -h <your_server_address> -p <port> -U guest_user -d ingo_benchmarks -c "SELECT * FROM poseidon_benchmark;"
```

Ensure that your server's firewall and network settings allow for remote connections on port 5432.

Its also possible to connect using PostgreSQL UI clients.

### 7. Clean-up Process

To stop and remove the container, run:

```sh
docker-compose down
```

To remove the Docker volume and image, use:

```sh
docker volume rm zk_benchmark_data
docker rmi zk-benchmark-db
```


### References

- [Official Docker Compose Documentation](https://www.docker.com/blog/)
- [How to Use the Postgres Docker Official Image](how-to-use-the-postgres-docker-official-image/)

