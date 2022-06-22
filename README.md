# Installation

## Installing Python
You will need Python version 3.10 or greater to run this package. This version was chosen for its better support for type hints.
If you wish to use an earlier version of Python, you will need to remove the offending type hints.

### Binary Download
Various binaries for different operating systems are available at https://www.python.org/downloads/.

### Installing on MacOS
If you don't already have Homebrew installed, use the command below to install it or visit http://brew.sh.
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Update Homebrew by running:
```bash
brew update
```

If you don't already have a version of Python 3, install it with the command:
```bash
brew install python3
```

Otherwise update your existing python3 installation
```bash
brew upgrade python3
```

Check that the Python installation is the correct version.
```bash
python3 --version
```

At the time of writing this README, the latest version in brew was 3.9. In this case you will instead need to run:
```bash
brew install python@3.10
```

You will need to have this at the beginning of your PATH. For the purposes of setup, you only need to do this for this session.
Check the output from `brew install` for the correct path to the newly installed Python version and add it to your PATH
```bash
exportPathCMD='export PATH="/usr/local/opt/python@3.10/bin:$PATH"'
eval $exportPathCMD
```

If you want to always use this version of Python for new sessions, run:
```bash
echo $exportPathCMD >> ~/.bash_profile
```

### Installing On Linux
```bash
sudo apt update
sudo apt install python3.10
```
### Installing on Windows
See the section [Binary Download](#binary-download). This is the preferred way to install on Windows.

## Clone the repository
Navigate to the desired working directory and clone the git repo.
`cd` into the newly cloned directory.
```bash
git clone git@github.com:mattbaumgartner/pyVersium.git
cd PyVersium
```

## Setting Up The Virtual Environment
If you followed the directions above, you should see that your Python 3 version is 3.9 or greater.
If not you may have to adjust your _$PATH_ variable to point to the correct version. The instructions below are for Mac/Linux.
For detailed instructions on using virtual environments on Windows, see
[this page](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Check that you have the correct version of Python by running the command below:
```bash
python3 --version
```
If you haven't already, `cd` into the cloned directory. You should see this README document in the directory.

### Creating the environment
Create the virtual environment by running the following command:
```bash
python3 -m venv virtualenv
```
### Activating the environment
Now activate the environment:
```bash
source virtualenv/bin/activate
```
You will need to activate this environment whenever you want to use this package. This will keep your main Python distribution
clean and make reinstallation a breeze

### Installing packages in the environment
With the virtual environment activated, run:
```bash
python3 -m pip install .
```

### Deactivating the environment
When you are finished you can deactivate the current environment to return your __PATH__ to its original state: 
```bash
deactivate
```

# Usage

## CLI 
Activate your virtual environment as described [above](#activating-the-environment).
The package installation will have added two commands to your path: `append` and `model`
You can get help by running
```bash
append --help
model --help
```
## Config File
In addition to options passed directly to the CLI, you can specify a config file in JSON format. The `-c FILEPATH`
or `--config FILEPATH` option can be used to supply a config file in JSON format. All options available in the CLI,
except `--config`, can be supplied via the config file with key-value pairs. When the same option is provided both via the command line
and a config file, the command line option will take priority. This allows you to make small changes between runs without
needing to edit the config file directly. 

You can specify an option in the config file by taking the long form of the CLI option, removing the initial `--` and replacing all intermediate `-` 
with underscores. For example,

```bash
--input my/input.txt --output my/output.txt --log-file my/log.txt
```
would become 

```json
{
  "input": "my/input.txt",
  "output": "my/output.txt",
  "log_file": "my/log.txt"
}
```

### Specifying Subcommands in the Config File
You can specify options for specific subcommands such as `train` and `score` by giving the subcommand as the key and
a JSON object as the value. For example, you can provide a different input and log file for training and scoring to avoid
having to supply these to the command line everytime. This can also be used to supply options that are only applicable to
a specific subcommand such as ___chunksize___ for `score` or ___label___ for `train`.

Below is an example of a config file utilizing the `score` and `train` subcommands:
```json
{
  "log_level": "INFO",
  "model_dir": "my/model_dir",
  
  "score": {
    "input": "my/score_input.txt",
    "output": "my/score_output.txt",
    "chunksize": 5000,
    "log_file": "my/score_log.txt"
  },
  
  "train": {
    "input": "my/train_input.txt",
    "log_file": "my/train_log.txt",
    "label": "my_label"
  }
}
```

### Output Files
Output files can be given special formatting to control their naming. When the name of an output file contains `$@`, these
characters will be substituted with the basename of the input file minus the extension. For example:

if the input is:
```
path/to/input/my_file.txt
```
and the output file is:
```
path/to/output/$@_appended.txt
```

Then the output of the program will be written to:
```
path/to/output/my_file_appended.txt
```
