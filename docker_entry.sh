#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${OUTPUT_DIR:-/app/out}"
chmod -R 777 "${OUTPUT_DIR:-/app/out}" || true

exec micromamba run -n base python -m src.main


