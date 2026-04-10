#!/usr/bin/env bash

INPUT_FILE="life-expectancy.csv"
OUTPUT_DIR="data-per-country"

mkdir -p "$OUTPUT_DIR"

# Get unique country names (skip header)
cut -d',' -f1 "$INPUT_FILE" | tail -n +2 | sort | uniq | while read -r country; do
    echo "Processing $country"

    # Extract all rows for this country
    awk -F',' -v c="$country" '$1 == c' "$INPUT_FILE" > "$OUTPUT_DIR/${country}.csv"

done
