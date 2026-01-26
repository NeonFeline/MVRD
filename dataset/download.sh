#!/bin/bash

# Define variables
URL="https://database.lichess.org/lichess_db_eval.jsonl.zst"
DEST_DIR="data"
FILE_NAME="lichess_db_eval.jsonl.zst"

# Create the data directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Starting download of Lichess evaluation dataset..."

# Download the file
# -c: continues a partial download
# -P: specifies the directory prefix
# --show-progress: gives a better status bar
wget -c "$URL" -P "$DEST_DIR" --show-progress

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download complete: $DEST_DIR/$FILE_NAME"
else
    echo "Download failed. Please check your internet connection or the URL."
    exit 1
fi
