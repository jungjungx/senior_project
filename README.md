
# Senior Project 2024
This is my repository for everything relating to the Senior Project at University of Miami 2024

Here are what the files do:

ble looker.py is the python script. It does this:
1. scan for bluetooth (including BLE) devices
2. If it finds one that matches the MAC address (of the adafruit chip), it connects
3. once connected, goes into a loop

I also included a folder, senior_project that includes a slightly modified code for the adafruit. To make it work, you will need to add it to the examples folder.
The modified adafruit is code is necessary because the python script is made in a way that only receives data while the adafruit code only sends data.
'

### NEW STUFF

I have added the test machine learning model at folder "machine learning practice"

You will see multiple folders and files, heres what they all do:
1. **database** - directly from psyonet https://physionet.org/content/emgdb/1.0.0/ <br>
The model was trained off of this data.
2. **test_database** - heres the folder containing the "test" data that will be tested by the trained model. NOTE: Currently set up where I simply changed the name of the database (and inside the header file) so the trained model can only recognize the type of signal through the data only.
3. **psyonet_dataset**- python file that takes in all the data in the database
4. **EMGClassifier** - python file that trains the model and outputs:
5. **EMG_trained.pth** - trained model file
6. **EMG test**- python file that runs the test using the pth file

#### Important things to note
The data is set up to read both from the .dat file and the .hea file (idk if that might be a problem) <br>
