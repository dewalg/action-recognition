# Action_Recognition
CSE291D Group <br>
Dewal Gupta, Mithun Chakaravarrti, and Patrick Hayes <br>
UCSD Winter 2018

## Organization
Code is organized into 4 main categories: download, utils, models and config.

### Download
Contains files that are related to the downloading and using of the kinetics dataset. A great
resource is the official activitynet github repository which contains official code
to help download the large dataset. In our directory, there's a download.py script that reads
a CSV file to determine which files to download and from where. It also uses ffmpeg to clip
these mp4 files to their respective 10 second windows. Please see activitynet on how to use
and the dependencies. 

### Utils
The utils directory contains preprocessing code that helps transform the mp4 files into 
relevant jpg files. These jpg files are directly read by the tensorflow model and turned
into tensors at that time. It also contains a script to help generate the labels and 
filepath text files for input into the system. 

### config
This directory holds mainly one config.ini file which contains a variety of information
used by our models. This includes save diretories, input/data directories, hyperparameters, etc.
It also contains text files for the set of labels and the filepaths for the videos. 

### models
This is the most important directory since it contains the files for our models that we have
used and trained using "train.py" (also in the directory). There are many different models with slight
variations that were used for experimentation. This also contains a "pipeline.py" file which is reponsible
for creating a DataSet pipeline using the tensorflow API to feed the videos into the model for training.

