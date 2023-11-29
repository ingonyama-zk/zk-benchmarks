# Example project: how to insert benchmarks into database

Workflow:

1. Create metadata json from your benchmark function
2. Run benchmarks and save results to json
3. Run rust app:
	- read metadata and bechmarks json
	- insert into database

`cargo install cargo-criterion`
`cargo criterion --message-format=json > /tmp/criterion.json`


