#!/bin/bash

# exit on error
set -e
# enable debug mode
set -x

# load GITHUB_ACCESS_TOKEN from .env file
source ./.env

# check if the GITHUB_ACCESS_TOKEN variable is defined
if [[ -z "${GITHUB_ACCESS_TOKEN}" ]]; then
  echo "Error: GITHUB_ACCESS_TOKEN is not defined. Define it in the .env file."
  exit 1
fi

# remove previous runner, if any
docker rm my-github-runner || true
docker rmi github-runner || true
docker system prune -a --force --volumes

# obtain new runner tocken
token=$(curl --silent --show-error -X POST \
-H "Authorization: token ${GITHUB_ACCESS_TOKEN}" \
-H "Accept: application/vnd.github.v3+json" \
https://api.github.com/repos/ingonyama-zk/zk-benchmarks/actions/runners/registration-token \
| jq -r '.token')

echo "Received token: $token"

docker build --build-arg RUNNER_TOKEN=$token -t github-runner .
# Uncomment one of the following lines to run the runner in detached or interactive mode
#docker run -d --gpus all --name my-github-runner github-runner ./run.sh --once
#docker run -it --rm --gpus all --name my-github-runner github-runner