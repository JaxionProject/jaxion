import importlib.resources
import json


def print_parameters(params):
    print(json.dumps(params, indent=2))


def set_up_parameters(user_overwrites):
    # first load the default params
    with importlib.resources.open_text("jaxion", "params_default.json") as f:
        params = json.load(f)

    # go down to lowest level in dict and eliminate meta-data
    params = _eliminate_metadata(params)

    # update default values with user-supplied overwrites
    params = _update_dicts(params, user_overwrites)

    # XXX update git_hash/library version

    return params


def _eliminate_metadata(params):
    for key, value in params.items():
        if isinstance(value, dict):
            if "default" not in value:
                _eliminate_metadata(value)
            else:
                params[key] = value["default"]

    return params


def _update_dicts(orig_dict, new_dict):
    for key, value in new_dict.items():
        if (
            key in orig_dict
            and isinstance(orig_dict[key], dict)
            and isinstance(value, dict)
        ):
            _update_dicts(orig_dict[key], value)
        else:
            if key in orig_dict:
                if not isinstance(value, dict):
                    orig_dict[key] = value
                else:
                    raise ValueError(
                        f"Value: {value} for parameter key: {key} must be a value, not dict"
                    )
            else:
                raise KeyError(f"Unknown parameter key: {key}")
    return orig_dict
