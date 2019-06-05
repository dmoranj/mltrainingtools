import numpy as np


GENERATOR_FUNCTIONS = {
    "smallfloat": lambda base, range, number: list(np.power(10, -(range*np.random.rand(number) + base))),
    "integer": lambda base, range, number: list(np.floor(base + range*np.random.rand(number)).astype(int))
}


def generate_metaparameters(number, definition):
    results = {}

    for key, parameter in definition.items():
        generator_fn = GENERATOR_FUNCTIONS.get(parameter['type'], parameter['default'] * number)
        results[key] = generator_fn(parameter['base'], parameter['range'], number)

    return results

