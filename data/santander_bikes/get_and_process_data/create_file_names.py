"""
A Python file to create a .txt file of all files to be downloaded using get_data.sh
"""


from datetime import datetime, timedelta
import os

print(os.getcwd())
# Initialize variables
n = 143
date_format = "%d%b%Y"
start_date = datetime.strptime("02Jan2019", date_format)
end_date = datetime.strptime("04Jan2022", date_format)
output_file = "data/santander_bikes/file_names.txt"
repeated_n = 246
repeated_once = False

# Open the file for writing
with open(output_file, "w") as file:
    current_date = start_date
    while current_date <= end_date:
        date1_str = current_date.strftime(date_format)
        date2 = current_date + timedelta(days=6)
        date2_str = date2.strftime(date_format)
        
        # Write the string to the file
        file.write(f"{n}JourneyDataExtract{date1_str}-{date2_str}.csv\n")
        
        # Increment the date and the number
        current_date += timedelta(days=7)
        
        # Handle the repetition of 246
        if n == repeated_n and not repeated_once:
            repeated_once = True  # Set flag to repeat 246 once
        else:
            n += 1

print(f"File {output_file} generated successfully.")

