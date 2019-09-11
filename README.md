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

This module contains some utilities related mainly with Computer Vision problems.

### IoU(x, y, h, w, x_h, y_h, h_h, w_h))

Computes the Intersection over Union of the two rectangles given as parameters. Rectangles are expressed as
tuples of four numbers: (x, y) representing the top left corner of the rectangle and (h, w) representing the
height and width of the rectangle. IoU operation is simmetric (the order of the rectangles doesn't change the
end result).

## dnnutils

### load_pretrained(num_classes, backbone, finetune=False, remove_linear=True, finetune_skipping=None)

Loads and initializes a pretrained CNN model for feature extraction with the output size given by `num_classes`, with 
the selected  backbone. Currently available backbones are: "RESNET101" and "VGG16". By default, all layers are freezed
and won't be finetuned during training. This behavior can be overriden using the `finetune` flag: setting this flag to 
true will allow finetuninng of all layers above the Nth (where N depends on the backend, and can be overriden with the
`finetune_skipping` parameter).

For those models using Fully Connected layers for classification, those layers can be removed and substituted by an
AveragePooling layer using the `remove_linear` flag. 

This function returns two values: the input size required for the feature

### create_lr_policy(milestones, multipliers)

Creates a Learning Rate Policy function that produces a learning rate multiplier based on a sequence of epoch number
milestones. This function can be used along with PyTorch's LambdaLR scheduler to create a learning rate schedule.

The following excerpt shows an example of schedule creation:

```python
scheduler = LambdaLR(optimizer, lr_lambda=create_lr_policy([10, 50, 80], multipliers=[1, 10, 1, 0.1]))
```

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

