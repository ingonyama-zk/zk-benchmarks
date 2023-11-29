
\connect ingo_benchmarks

-- Create a guest_user with read-only privileges and no password

CREATE USER guest_user WITH
    LOGIN -- Allows the user to log in
    NOSUPERUSER -- Ensures the user is not a superuser
    NOCREATEDB -- Prevents the user from creating databases
    NOCREATEROLE -- Prevents the user from creating roles
    INHERIT -- Allows the user to inherit roles
    NOREPLICATION -- Prevents the user from replicating data
    CONNECTION LIMIT -1; -- Sets an unlimited connection limit

-- Grant read-only privileges to the user on a specific database 'ingo_benchmarks'
GRANT CONNECT ON DATABASE ingo_benchmarks TO guest_user;
GRANT USAGE ON SCHEMA public TO guest_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO guest_user;

-- Revoke other privileges
-- REVOKE ALL PRIVILEGES ON DATABASE ingo_benchmarks FROM guest_user;
-- REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM guest_user;