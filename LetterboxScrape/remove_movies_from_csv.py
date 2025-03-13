import csv

# Define file names
csv_file = './unique_film_ids_full.csv'          # Input CSV file
remove_file = './less_25_film_ids.csv'  # File with movie names to remove
output_file = 'filtered_movies.csv'   # Output CSV file

# Step 1: Read the movie names to remove into a set
with open(remove_file, 'r') as f:
    movies_to_remove = set(line.strip() for line in f)

# Step 2: Read and filter the CSV file
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Read the header ('filmd_ids')
    # Filter rows: keep only those not in movies_to_remove
    rows = [row for row in reader if row[0] not in movies_to_remove]

# Step 3: Write the filtered data to a new CSV file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write the header
    writer.writerows(rows)   # Write the filtered rows

print(f"Filtered CSV has been saved to {output_file}")
