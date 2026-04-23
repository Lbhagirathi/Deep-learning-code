#!/usr/bin/env bash

INPUT_FILE="life-expectancy.csv"

# Extract country column, skip header, count occurrences
cut -d',' -f1 "$INPUT_FILE" | tail -n +2 | sort | uniq -c | \
awk '{print $2","$1}'

# cut -d',' -f1	extract country column
# tail -n +2	remove header
# sort	group identical countries
# uniq -c	count occurrences
# awk '{print $2","$1}'	format output as Country,Count

# -t','	CSV delimiter
# -k2	sort by column 2
# -n	numeric sort
# -r	reverse order (largest first)

# sort file.txt                 # sort lines alphabetically (ascending)
# sort -r file.txt              # reverse order (descending)
# sort -n file.txt              # numeric sort
# sort -nr file.txt             # numeric descending
# sort -u file.txt              # unique lines only

# sort -t',' -k1 file.csv       # sort by column 1
# sort -t',' -k2 file.csv       # sort by column 2
# sort -t',' -k2 -n file.csv    # numeric sort by column 2
# sort -t',' -k2 -nr file.csv   # numeric descending by column 2

# sort -t',' -k1,1 -k2,2 file.csv

# Meaning:

# Sort by column 1
# If equal → sort by column 2

# chmod +x get_counts.sh
# ./get_counts.sh | sort -t',' -k2 -nr > entries_per_country.csv

# cut -d',' -f1 life-expectancy.csv | tail -n +2 | sort | uniq -c | awk '{print $2","$1}'

# ls -l | sort -k5 -n          # smallest files first
# ls -l | sort -k5 -nr         # largest files first