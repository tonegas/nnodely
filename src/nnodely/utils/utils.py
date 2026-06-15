# This file contains utility functions for the nnodely library.

def _flatten_dict(d):
    # Flatten a dictionary of the form {key: value} where value can be a string or a dict of the same form.
    if not isinstance(d, dict):
        return d
    
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested = _flatten_dict(value)
            for nested_key, nested_value in nested.items():
                result[nested_key] = nested_value
        else:
            result[key] = value
    return result