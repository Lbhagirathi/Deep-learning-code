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

# awk → text processing tool used to filter and manipulate columns in files

# -F',' → sets the field separator to a comma
# This tells awk that the file is a CSV and columns are separated by ','

# -v c="$country" → creates an awk variable named c
# It stores the current country name coming from the bash loop

# '$1 == c' → condition
# $1 refers to the first column in the CSV file (Country column)
# This checks if the value in column 1 equals the country name stored in variable c

# "$INPUT_FILE" → the file being searched (life-expectancy.csv)

# If the condition ($1 == c) is true,
# awk prints that entire row

# > → redirect operator
# Sends the output to a new file instead of printing to the terminal

# "$OUTPUT_DIR/${country}.csv" → output file path
# Creates a CSV file named after the country inside the data-per-country folder

# Overall:
# This command extracts all rows belonging to one country
# and saves them into a separate CSV file named after that country

# Extract row 5	sed -n '5p' file.csv
# Extract row 5 (awk)	awk 'NR==5' file.csv
# Extract header	head -n 1 file.csv
# Extract rows 2–10	sed -n '2,10p' file.csv