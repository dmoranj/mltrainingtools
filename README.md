# Python Machine Learning Training toolset 

This project will contain the utility functions I'll be creating for PyTorch ML projects.

# Installation

This module is still in development, and thus it has been installed in the PyPy test
repository. It can be locally installed with the following command:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps mltrainingtools-dmoranj
```


Once installed locally, it can be imported as usual with:

```python
import mltrainingtools
```

# Modules

## cvutils

## cnnutils

## metaparameters

This module contains some functions to deal with metaparameter search scenarios.

### generate_metaparameters(number, definition, static=False)

Given a metaparameter set definition and a number, this function generates a random collection
of values for the parameters, according to the specifications.

The metaparameters can be defined with a Python dict, where each of the keys is one metaparameter.
The definition of each parameter **must** contain the following fields:

* *base*: base value for the parameter, all the parameters randomly drawn will be higher than this number.
* *range*: range of variation of the parameters. All the parameters will fall in the interval *(base, base + range)*
* *type*: a string indicating the parameter type. There are currently two types of supported parameters:
    * *integer*: integer parameters will be drawn uniformily from the selected interval.
    * *smallfloat*: random draws for small float parameters are drawn from exponentially from the interval *(10^(-base - range), 10^(-base))* 
      This parameters are meant to be used in small values that need logarithmic search (e.g.: the learning rate).
* *default*: all parameters can be provided with a default value. This value is used to force particular values of the
  metaparameters using the *static* parameter of the function (e.g.: for debugging purposes).
  
The following excerpt show an example of a valid metaparameter definition:

```text
INTEGER_DEF_1 = {
  'Integer1':
        {
            'base': 10,
            'range': 90,
            'default': 5,
            'type': 'integer'
        },
  'SmallFloat1':
       {
           'base': 1,
           'range': 4,
           'default': 0.0,
           'type': 'smallfloat'
       }
}
```

## cmdlogging

This module contains some simple logging utilities for command line experiments.

### section_logger(level=0)

This function generates a logger that can be invoke to output text to the console. The logs output the amount of time
from the creation of the logger along with the provided text. The logs are structured in a hierarchycal way (the *level*
of the hierarchy can be passed as a parameter). The logger will add different indentations and line indicators for each
different level.


# Publishing

The module can be published as a regular PIP package, using the following commands:

* From the directory where the `setup.py` is located, execute the following command to generate
  the binary artifacts under `dist/`:

```bash
python3 setup.py sdist bdist_wheel
```

* From the root directory, execute the following command to upload a new version:

```bash
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

The package should now be ready to be imported (or updated).

