import pandas as pd

def convert_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Convert the single column into a row
    row_data = df.iloc[:250, 0].astype(str)

    # Prepend '0' to the beginning of the row data
    row_data_with_zero = ['0'] + row_data.tolist()

    # Write the row data into a new CSV file
    with open(output_file, 'w') as f:
        f.write(','.join(row_data_with_zero))
# Example usage
input_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\VoltageTest.csv"
output_file = "C:\\Users\\Jakeeer\\git\\senior_project\\machine learning p3\\VoltageTest_compiled.csv"
convert_csv(input_file, output_file)
