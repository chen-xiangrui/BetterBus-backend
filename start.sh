#!/bin/bash

cd "$(dirname "$0")"
uvicorn detect:app --uds=/tmp/uvicorn.sock