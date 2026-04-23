#!/bin/bash

file="/nfscommon/common/bedartha/quiz3/question1b.csv"

# Count lines containing "United Kingdom"
count=$(grep -c "United Kingdom" "$file")

echo "$count"