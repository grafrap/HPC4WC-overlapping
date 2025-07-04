import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# function for data extraction
def extract_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if "###" in line:
                line = line.replace("###", "")
                parts = line.strip().split(",")
                try:
                    time = int(parts[0].strip())
                    size = int(parts[1].strip())
                    num_streams = float(parts[2].strip())
                    data.append((threads, mesh, time))
                except Exception as e:
                    print(f"Skipping line (parsing error): {line}")
    return pd.DataFrame(data, columns=["Time", "Size", "NUM_STREAMS"])


# Take filename as argument
if len(sys.argv) < 2:
    print("Usage: python analyze_output.py <output_file.out>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Reading from: {filename}")
data = extract_data(filename)

# Save to CSV for future analysis
output_csv = filename.replace(".out", ".csv")
data.to_csv(output_csv, index=False)