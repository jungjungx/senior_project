import pandas as pd

def convert_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Convert the single column into a row
    row_data = df.iloc[:, 0].head(1000).tolist()

    # Write the row data into a new CSV file
    with open(output_file, 'w') as f:
        f.write(','.join(map(str, row_data)))

# Example usage
input_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\VoltageTest.csv"
output_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p2\\VoltageTest_compiled.csv"
convert_csv(input_file, output_file)
