import csv

# Define the input and output file paths
input_file = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\emg_csv.csv'
output_file = 'C:\\Users\\Jakeeer\\git\\senior_project\\machine learning practice\\modded_emg_csv.csv'

# Open the input CSV file for reading and the output CSV file for writing
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Iterate over each row in the input CSV file
    for row in reader:
        # Transform each value in the row according to the formula
        transformed_row = [(float(value) / 1023.0) * 3.3 for value in row]
        
        # Write the transformed row to the output CSV file
        writer.writerow(transformed_row)

print("Transformation complete.")
