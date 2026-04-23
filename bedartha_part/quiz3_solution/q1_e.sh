#!/bin/bash

# List files sorted by modification time, pick the first
latest=$(ls -t | head -n 1)

echo "$latest"