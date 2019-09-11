import numpy as np


GENERATOR_FUNCTIONS = {
    "smallfloat": lambda base, range, number: list(np.power(10, -(range*np.random.rand(number) + base))),
    "integer": lambda base, range, number: list(np.floor(base + range*np.random.rand(number)).astype(int))
}


def to_list(metaparameters):
    result = []
    L = len(metaparameters[list(metaparameters.keys())[0]])

    for i in range(L):
        parameter = {}
        for key in metaparameters.keys():
            parameter[key] = metaparameters[key][i]

        result.append(parameter)

    return result


def generate_metaparameters(number, definition, static=False):
    results = {}

    for key, parameter in definition.items():
        if static:
            results[key] = [parameter['default']] * number
        else:
            generator_fn = GENERATOR_FUNCTIONS.get(parameter['type'], [parameter['default']] * number)
            results[key] = generator_fn(parameter['base'], parameter['range'], number)

    return results

