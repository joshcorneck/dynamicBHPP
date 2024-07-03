## Get data
mkdir ../csv_files

cat online-networks-change-points/data/santander_bikes/get_and_process_data/file_names.txt | while read line 
do
   curl https://cycling.data.tfl.gov.uk/usage-stats/$line --output online-networks-change-points/data/santander_bikes/csv_files/$line
done
