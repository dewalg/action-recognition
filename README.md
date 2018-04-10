# Action_Recognition
Dewal Gupta, Mithun Chakaravarrti<br>
Dr. Manmohan Chandraker, UCSD

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

### References
Some of the models and code are original, while some are based on previous work:
1. i3d models are based on work as described in <a href="https://arxiv.org/abs/1705.07750">Quo Vadis</a> by Carreira et al. The original repository with their code can be found <a href="https://github.com/deepmind/kinetics-i3d">here</a>.
2. s3d models are based on the work done by Xie et al. in <a href="https://arxiv.org/abs/1712.04851">Rethinking Spatiotemporal Feature Learning For Video Understanding</a>. The authors do not provide any public code implementing their models. 
