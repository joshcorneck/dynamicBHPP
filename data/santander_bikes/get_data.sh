## Get training data
mkdir training
cat file_names_training.txt | while read line 
do
   curl https://cycling.data.tfl.gov.uk/usage-stats/$line --output training/$line
done

## Get test data
mkdir test
cat file_names_test.txt | while read line
do
   curl https://cycling.data.tfl.gov.uk/usage-stats/$line --output test/$line
done
