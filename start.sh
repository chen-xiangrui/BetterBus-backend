#!/bin/bash

cd "$(dirname "$0")"
/home/ubuntu/.local/lib/python3.12/site-packages/uvicorn detect:app --host=0.0.0.0 --workers=4