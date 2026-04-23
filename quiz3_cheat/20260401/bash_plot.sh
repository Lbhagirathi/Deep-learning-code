#!/usr/bin/env bash

INPUT_DIR="data-per-country"
OUTPUT_DIR="plots-per-country"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.csv
do
    echo "Plotting $file"
    python plot_life_expectancy.py "$file" "$OUTPUT_DIR"
done

# mkdir -p plots-per-country