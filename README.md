# 🎵 STM: Sound Topic Modeling
Largely based on the excellent work in [hb-song-analysis](https://github.com/tbergama/hb-song-analysis)

This project is a Python-based sound topic modeling tool that uses the [rost-cli tool](https://gitlab.com/warplab/rost-cli) 
to analyze audio data. 

There are many parameters that can be adjusted to tune the model to the data. 
Default parameters are set to work well with the MBARI MARS hydrophone data, but can be adjusted.
All parameters are in the `conf.py` file.

To use, place your audio data in the `data/` directory, 
and set the `target_file` variable in `conf.py` to the name of an audio file you
would like to visualize the output on.

##  Getting Up and Running

🐳 Option 1: ROST via Docker (Recommended)

Build the Docker image from the project root:

```shell
docker build -t rost-cli -f DockerfileROST .
```

Then run the container to test with the `topics.refine.t` command:

```shell
docker run -it --rm rost-cli topics.refine.t --help
```

You should see the help message for the `topics.refine.t` command, which indicates that the ROST CLI is working correctly.
```text
(venv) docker run -it --rm rost-cli topics.refine.t --help                                                                                                                                                                                                                 (stm) 12:35:46
Topic modeling of data with 1 dimensional structure.:
  --help                                help
  -i [ --in.words ] arg (=/dev/stdin)   Word frequency count file. Each line is
                                        a document/cell, with integer 
                                        representation of words. 
  --in.words.delim arg (=,)             delimiter used to seperate words.
  --out.topics arg (=topics.csv)        Output topics file
  --out.topics.ml arg (=topics.maxlikelihood.csv)
                                        Output maximum likelihood topics file
  --out.topicmodel arg (=topicmodel.csv)
                                        Output topic model file
  --in.topicmodel arg                   Input topic model file
  --in.topics arg                       Initial topic labels

```

Make sure the docker engine is running through python with the command:

```shell
python -c "import docker; print(docker.__version__)"  
```

You should see the version of the docker package installed in your Python environment, e.g.

```text
7.1.0
```

if not running, run the command:
```shell
pip3 install docker
```
#### Option 2: ROST Local
Clone the [rost-cli repository](https://gitlab.com/warplab/rost-cli) 

```shell
git clone https://gitlab.com/warplab/rost-cli
```

and follow the installation instructions found in their README.

Once ROST is installed, change the `rost_path` variable in conf.py to be the path to /rost-cli/bin/ directory
on your machine.

### Installing Python Dependencies

Install with the package manager your prefer.

With conda,
```shell
conda env create
```
With virtualenv, `pip install virtualenv` then
```angular2html
virtualenv --python=python3.10 venv
source venv/bin/activate
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
On top is a spectrogram of a ~50-second segment of the target file specified in conf.py. On bottom
is a stacked bar plot of topic probabilities over documents, where documents represent small increments
of time.

#### Quick Notes
* If the modules are run individually, they should be run in the order that they are run in run_model.sh.
each is dependent on the output of the last.
* If visualize.py raises an error saying that Python was not installed as a framework, try the 
solution [here](https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv)

## References
* ROST: https://gitlab.com/warplab/rost-cli
