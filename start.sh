#!/bin/bash

cd "$(dirname "$0")"
uvicorn detect:app --host=0.0.0.0 --workers=4