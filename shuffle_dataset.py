import zstandard as zstd
import io
import random
import os
import sys

def main():
    input_file = "dataset/data/lichess_db_eval.jsonl.zst"
    output_file = "dataset/data/lichess_db_eval_shuffled.jsonl.zst"
    
    print(f"Loading {input_file} into memory...")
    lines = []
    
    dctx = zstd.ZstdDecompressor()
    with open(input_file, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            # Wrap in BufferedReader to read lines efficiently
            br = io.BufferedReader(reader)
            while True:
                line = br.readline()
                if not line:
                    break
                lines.append(line)
                if len(lines) % 1000000 == 0:
                    print(f"Loaded {len(lines)} lines...")
                    
    print(f"Total lines loaded: {len(lines)}")
    print("Shuffling lines...")
    random.seed(42)
    random.shuffle(lines)
    
    print(f"Saving shuffled dataset to {output_file}...")
    cctx = zstd.ZstdCompressor(level=3)
    with open(output_file, 'wb') as f:
        with cctx.stream_writer(f) as writer:
            for i, line in enumerate(lines):
                writer.write(line)
                if (i + 1) % 1000000 == 0:
                    print(f"Written {i + 1} lines...")
                    
    print("Done! You can now replace the old dataset with the shuffled one:")
    print(f"mv {output_file} {input_file}")

if __name__ == "__main__":
    main()
