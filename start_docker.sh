#!/usr/bin/env bash
# Build and start the full stack (requires a filled-in .env; see .env.example).
set -e
docker compose up --build
