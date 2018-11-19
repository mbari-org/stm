# hb-song-analysis: Topic Modeling for Humpback Whale Song

## Getting Up and Running
### Compatability
Please note that currently this project is not compatable with Windows operating systems.
### Installing ROST
This project wraps the Realtime Online Spatiotemporal Topic Modeling (ROST) library from WHOI's WARPLab.
In order to run this project you will need to clone their [repository](https://gitlab.com/warplab/rost-cli) and follow
the installation instructions found in their README.  
Once ROST is installed, change the `rost_path` variable in conf.py to be the path to /rost-cli/bin/ directory
on your machine.
### Installing Python Dependencies
The dependencies of this project are listed requirements.txt. The code below will install these
requirements in a virtual environment. This requires virtualenv, if you don't have it just run
a quick `pip install virtualenv`.
```angular2html
$ virtualenv --python=python3.6 venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```
### Running the Project
Once all dependencies are installed, the project can be run with `(venv) $ ./run_model.sh`, which will run all
project modules using the parameters specified in conf.py. 
Modules can be run individually with `(venv) $ python module_name.py`. For example, the stft module
can be run individually with `(venv) $ python stft.py`. You may want to run the modules
individually to test different parameters of that module without rerunning the previous modules.  
A successful run will produce a plot similar to the following:
![topic bar plot](./img/stacked_bar.png)
On top is a spectrogram of a ~50 second segment of the target file specified in conf.py. On bottom
is a stacked bar plot of topic probabilities over documents, where documents represent small increments
of time.

#### Quick Notes
* If the modules are run individually, they should be run in the order that they are run in run_model.sh.
each is dependant on the output of the last.
* If visualize.py raises an error saying that Python was not installed as a framework, try the 
solution [here](https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv)

## References
* ROST: https://gitlab.com/warplab/rost-cli