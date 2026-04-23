#!/bin/bash

dir="/nfscommon/common/bedartha/quiz3/question1d/"

for file in "$dir"/file_*.txt; do
    base=$(basename "$file")
    newname=$(echo "$base" | sed 's/^file_/oldfile_/')
    mv "$file" "$dir/$newname"
done