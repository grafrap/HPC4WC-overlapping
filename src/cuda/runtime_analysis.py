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
                    num_reps = int(parts[0])
                    size = int(parts[1])
                    num_streams = int(parts[2])
                    time = float(parts[3])
                    data.append((num_reps, size, num_streams, time))
                except Exception as e:
                    print(f"Skipping line (parsing error): {line}")
    return pd.DataFrame(data, columns=["Num_Repetitions","Size", "Num_Streams", "Time"])

    



# Take filename as argument
if len(sys.argv) < 2:
    print("Usage: python analyze_output.py <output_file.out>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Reading from: {filename}")
data = extract_data(filename)

# Set pandas display options to show scientific notation
pd.set_option('display.float_format', '{:.6e}'.format)

print(data)

# Save to CSV for future analysis
csv_path = os.path.splitext(filename)[0] + ".csv"
print(f"Saving CSV to: {csv_path}")

# First save the CSV (this will preserve full precision)
# Format the Time column in scientific notation for CSV output
data_formatted = data.copy()
data_formatted['Time'] = data_formatted['Time'].apply(lambda x: f'{x:.6e}')
data_formatted.to_csv(csv_path, index=False)

# Then read it back and print a few lines to verify
check_df = pd.read_csv(csv_path)
print("Reloaded CSV preview:")
print(check_df.head())