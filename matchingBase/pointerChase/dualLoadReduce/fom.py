# Define the path to your input file
input_file_path = 'FOM.res'

# Open the input file for reading
with open(input_file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Strip the newline character from the end of the line and split the line by commas
        parts = line.strip().split(',')
        
        # Extract the relevant parts of the line for calculation
        graph_name = parts[0]
        first_number = float(parts[1])
        second_number = float(parts[2])
        third_number = float(parts[3])
        fourth_number = float(parts[4])
        
        # Perform the calculations: divide the first number by the second, and the third by the fourth
        first_division_result = first_number / second_number if second_number != 0 else "Undefined"
        second_division_result = third_number / fourth_number if fourth_number != 0 else "Undefined"
        
        # Print the results to stdout
        print(f"{graph_name}: First Division = {first_division_result}, Second Division = {second_division_result}")

# Note: This script includes a simple check to avoid division by zero errors.
