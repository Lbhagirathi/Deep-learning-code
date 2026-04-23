#!/bin/bash

input="/nfscommon/common/bedartha/quiz3/question1c.csv"
output="question1c.csv"

# Assuming country is in one column (e.g., column 1)
cut -d',' -f1 "$input" | sort | uniq > "$output"

echo "Unique countries saved to $output"