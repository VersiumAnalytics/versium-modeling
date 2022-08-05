# About
pyVersium is a Python library and set of command line tools for the creation of lead scoring pipelines. The toolset
simplifies the process of appending data across multiple APIs and then building lead scoring models that leverage both
the original and appended data points.

# Installation

## Installing Python
You will need Python version 3.10. This version was chosen for its better support for type hints.
If you wish to use an earlier version of Python, you will need to remove the offending type hints.

You may be able to use a version higher than 3.10. This depends largely on whether Numpy supports that version of Python.

### Conda Installation
Conda is a Python package and environment management system and is the recommended way for installing Python and creating a virtual environment. Unless you are experienced with
Python package management, this should be your first choice for setting up your Python environment. Miniconda is one 
distribution of Conda that a
light-weight version of Conda that leaves out the bloatware packaged with the full  Anaconda. You will need to pick out the 
appropriate version of Miniconda based on your OS and system architecture at https://repo.anaconda.com/miniconda/. Copy
the link address of the version that best matches your scenario and substitute it in place of the URL in the following 
command:

```bash
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh && sh ~/miniconda.sh
```

Continue following the instructions to install Conda. Once finished, continue setup by following instructions in the
[Creating a Conda Environment](#creating-a-conda-environment) section.

### Binary Download
Various binaries for different operating systems are available at https://www.python.org/downloads/.

### Homebrew (macOS)
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

Otherwise, update your existing python3 installation
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
Check the output from `brew install` for the correct path to the newly installed Python version and add it to your PATH.
Alternatively you can run the following to get the path to the installation:
```bash
brew list @python3.10
```
Now change your ***PATH*** to point to the installation.
```bash
exportPathCMD='export PATH="/usr/local/Cellar/python@3.10/3.10.4/bin/:$PATH"'
eval $exportPathCMD
```

If you want to always use this version of Python for all new sessions, run:
```bash
echo $exportPathCMD >> ~/.bash_profile
```

To continue setting up the virtual environment, head over to the [venv](#venv-virtual-environment) section.

### Installing On Linux
Sometimes  attempt to install Python using the built-in system package management tool:
```bash
sudo apt update
sudo apt install python3.10
```
Continue environment setup by following the instructions in the [venv](#venv-virtual-environment) section.

### Installing on Windows
For installing on Windows, see the [Conda Installation](#conda-installation) section. If for some reason this is not an
option, try the [Binary Download](#binary-download) method.

## Clone the repository
Navigate to the desired working directory and clone the git repo.
`cd` into the newly cloned directory.
```bash
git clone https://github.com/VersiumAnalytics/pyVersium.git
cd pyVersium
```

## Setting Up An Environment
If you followed the directions above, you should have installed either Python or Conda on your system. Depending on the 
steps you followed, you will need to create a virtual environment using Conda, or venv if you did not go with the Conda
installation method.

### Creating a Conda Environment
Create a new Conda environment with Python 3.10 or greater. You can name this environment whatever you want with 
the `--name` flag.
```bash
conda create --name pyVersium python=3.10
```
Now activate the environment and install the package.
```bash
conda activate pyVersium
pip install --upgrade pip
pip install .
```
You will need to activate your Conda environment anytime you want to run the pyVersium tools. You should only need to
activate the environment once per shell session unless you have changed to a different Conda environment.

When you are finished running your jobs, you can deactivate the environment by running:
```bash
conda deactivate
```

### Venv Virtual Environment
If you did not go with the Conda installation method, you will need to create a virtual environment using ***venv***.
Check that your Python 3 version is 3.10 by running the following:
```bash
python3 --version
```

If not you may have to adjust your _$PATH_ variable to point to the correct version or substitute `python3` in the 
commands with `python3.10` or `python@3.10` if you used ***Homebrew***. The instructions below are for Mac/Linux.
For detailed instructions on using virtual environments on Windows, see
[this page](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

Create the virtual environment by running the following command:
```bash
python3 -m venv virtualenv
```

This will create a new folder called *virtualenv* in the current directory. This folder contains an isolated version of
the Python 3.10 binary that can be used to install requirements specific to this project without affecting the entire system.

Now activate the environment:
```bash
source virtualenv/bin/activate
```
You will need to activate this environment whenever you want to use this package. This will keep your main Python distribution
clean and make reinstallation a breeze.

With the virtual environment activated, run:
```bash
python3 -m pip install .
```

When you are finished you can deactivate the current environment to return your __PATH__ to its original state: 
```bash
deactivate
```

# Usage

## Quickstart Example

### Dataset
This example will guide you through the process of model training and scoring using the CLI and a config file. A dataset
and a config file have been included in this project's ___examples___ directory. The dataset we will be using is the 
[Breast Cancer Wisconsin (Diagnostic) Data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
We will be predicting the beast cancer diagnosis (malignant or benign) given features extracted from a digitized image
of an FNA of a breast mass. These features describe the characteristics of cell nuclei present in the image.

### Activate the Environment
Begin by activating your environment. This will depend on the name you gave during environment creation:
```bash
conda activate pyVersium
```

You should see something like `(pyVersium)` at the beginning of the current line in the shell. Check that pyVersium has
been installed and that the `model` command is on your path by running:

```bash
model -h
```
If everything is set up correctly you should see a help message.

In the help message you can see that there are two 
subcommands to `model`: `train` and `score`. As their names suggest, the `train` subcommand is used for training a model
and the `score` subcommand is used for scoring data with a trained model. Both of these subcommands accept additional
optional flags via the command line that control their behavior. Every option available via the CLI can also be provided
via a config file.

### Setting Up the Config 
The config file included in the ___examples___ folder contains the following:
```json
{
  "model_dir": "breast_cancer_model",
  "label": "target",
  "delimiter": "\t",
  "label_positive_value": "1",
  "log_level": "INFO",
  "random_state": 12345,

  "train": {
    "input": "breast_cancer_train.tsv"
  },

  "score": {
    "input": "breast_cancer_test.tsv",
    "output": "breast_cancer_test_scores.tsv",
    "include_fields": ["target"]
  }
}
```

* __model_dir__: Defines the directory where the model files and reports will be saved
* __label__: The name of the label or target column in the data
* __delimiter__: The column separator character for the data. The file we are using is .tsv, so our delimiter is a tab.
* __label_positive_value__: The value of the positive class in our data.
* __log_level__: The amount of information to log from the job. Options are: DEBUG, INFO, WARNING, ERROR, CRITICAL
* __random_state__: This is a random seed for reproducing the same modeling runs.

The above options will apply to all commands. You can also provide options that are specific to a subcommand. For the 
`train` subcommand, we will be using the input file ***breast_cancer_train.tsv***. For the `score` subcommand we are 
using ***breast_cancer_test_scores.tsv*** as our input file. The `train` subcommand doesn't output anything to a separate
file since it saves all the modeling information in the ***breast_cancer_model*** directory, but `score` does. The scores
will be saved to ***breast_cancer_test_scores.tsv***. If we were to omit the `include-fields` option, all the columns
in the input file would be included in our scoring output. We are only interested in what the original label was, so we
set ***target*** as the only other column to include with the score.

The config file is a good place to store options that we want to use as defaults for all of our training and scoring runs.
We can override the options in the config file by passing the corresponding flag directly to the command line. This is 
useful for trying different configurations without modifying the config file directly.

### Training the Model
Everything we need to build the model has been included in the config file. From the root project directory, navigate to
the ***examples*** folder
```bash
cd examples
```
Now train the model:
```bash
model train -c breast_cancer_config.json
```
You will see the progress of model building output. You should 

### Viewing the Report
Once model building has completed, a report will be generated in the ***report*** folder of the model directory. In this
case it should be in `examples/breast_cancer_model/report/report.html`. Open this file in your browser to view the modeling
results.

From the ROC Curve it's clear that the model performed well on its validation data and that there is a strong relationship
between the nuclei characteristics and the diagnosis. The relationship is so strong that the vast majority of scores lie
below 20 or above 90 even after calibration and score normalization. This is a toy example and models will generally 
not perform this well, especially for lead scoring use cases.

### Tuning the Model
We didn't do any optimization when training the model. We can add the `--num-opt` parameter to automatically tune the
model parameters. This option controls the number of rounds of tuning.

```bash
model train -c breast_cancer_config.json --num-opt 13
```

We can also try limiting the features we include to see how it affects performance. For example, we could use the 
*smoothness* features. For this run we will also use a new model directory so that we can compare both results.

```bash
model train -c breast_cancer_config.json --num-opt 13 -m breast_cancer_model2 --regex-include-fields '.*smoothness.*'
```

### Scoring with the Model
Scoring with the model works just like training. Since we have already set up the config file, we can just run:
```bash
model score -c breast_cancer_config.json
```

To use the other model with only *smoothness* features, we just need to specify the other model directory. We should 
also name the output something different.

```bash
model score -c breast_cancer_config.json -m breast_cancer_model2 -o breast_cancer_test_scores_model2.tsv
```


## CLI 
Activate your virtual environment as described in [above](#setting-up-an-environment).
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

### Append Config
When using the `append` CLI, you can provide a series of query configurations to perform data appends in succession.
```json
{
  "query_configs": [
    {
      "url": "http://api.versium.com/v2/b2cOnlineAudience",
      "queries_per_second": 20,
      "n_connections": 100,
      "n_retry": 3,
      "retry_wait_time": 3,
      "timeout": 10,
      "params": {
        "cfg_maxrecs": 1
      },
      "headers": {
        "Accept": "application/json",
        "x-versium-api-key": "abcdefg-123456789-hijklmn"
      },
      "required_fields": [
        "firstname",
        "lastname",
        "address"
      ],
      "optional_fields": [
        "zip",
        "state",
        "city"
      ],
      "field_remap": {
        "firstname": "first",
        "lastname": "last"
      },
      "post_append_prefix": "Versium_",
      "post_append_suffix": "",
      "response_handler": "api2b_versium_com_q2"
    },
    {
      "url": "http://append.mylistgensite.com/append",
      "params": {
        "api_key":"123456789",
        "param2": "something"
      },
      "required_fields": [
        "email",
        "phone"
      ]
    }
  ]
}
```

#### Query Config Params
* __url__: API endpoint for the query
* __queries_per_second__: Maximum number of queries per second.
* __n_connections__: Maximum number of simultaneous connections while querying.
* __n_retry__: Number of times to retry a query if it fails
* __retry_wait_time__: Number of seconds to wait before retrying a query. This scales with the number of attempts. For example, if 
set to 3 the wait times for each attempt will be 0, 3, 6, 9, etc.
* __timeout__: Number of seconds to wait for a response before timing out.
* __params__: Additional parameters to include with each request.
* __header__: HTTP header to pass with each request.
* __required_fields__: Data fields that are required to be present. If a record is missing one of these fields, it will not be appended. 
* __optional_fields__: Data fields that are not required to be present but will be included in the request if available.
* __field_remap__: Mapping of field names found in the input to parameter names expected by the API.
* __post_append_prefix__: Prefixes all fields returned by the API with a string. 
* __post_append_suffix__: Suffixes all fields returned by the API with a string.
* __response_handler__: Name of function to call to handle the response from the API. Custom response handler functions
can be added to *pyversium.collect.response_handlers* to extract data from the response.


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

When using the **_--chunksize_** option it is possible to split output into multiple files rather than one large file. In this 
case, you can specify a format string within the output filename that controls the numbering of files. The format string
takes the form `{START_IDX:FORMAT_SPEC}` where _**START_IDX**_ denotes the start of numbering and **_FORMAT_SPEC_**
is the Python format specification mini-language 
(see [python documentation](https://docs.python.org/3/library/string.html#format-specification-mini-language)).

For example, to start numbering at 2 with zero-padding to 3 places we could do:
```
path/to/output_file_{2:03d}.txt
```
