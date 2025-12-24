#!/bin/bash
# Meteor watcher script emits random streak reports.
COLORS=(crimson violet gold teal)
index=$((RANDOM % ${#COLORS[@]}))
length=$((RANDOM % 42 + 3))
echo "Meteor streak ${COLORS[$index]} spans ${length} degrees."
