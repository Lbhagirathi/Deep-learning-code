#!/bin/bash

dir="/nfscommon/common/bedartha/quiz3/question1a/"

# Count all entries (files + directories, excluding . and ..)
count=$(ls -1A "$dir" | wc -l)

echo "$count"