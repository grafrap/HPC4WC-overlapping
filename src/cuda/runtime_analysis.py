import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os



# function for data extraction
def extract_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if "###" in line:
                line = line.replace("###", "")
                parts = line.strip().split()
                try:
                    size = int(parts[0])
                    num_streams = int(parts[1])
                    time = float(parts[2])
                    data.append((size, num_streams, time))
                except Exception as e:
                    print(f"Skipping line (parsing error): {line}")
    return pd.DataFrame(data, columns=["Size", "NUM_STREAMS", "Time"])

    



# Take filename as argument
if len(sys.argv) < 2:
    print("Usage: python analyze_output.py <output_file.out>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Reading from: {filename}")
data = extract_data(filename)
print(data)

# Save to CSV for future analysis
csv_path = os.path.splitext(filename)[0] + ".csv"
print(f"Saving CSV to: {csv_path}")

# Read it back and print a few lines
check_df = pd.read_csv(csv_path)
print("Reloaded CSV preview:")
print(check_df.head())
data.to_csv(csv_path, index=False)