import collections

from ruamel.yaml import YAML


class YParams:
    def __init__(self, yaml_fn, config_name):
        self.hparams = {}
        with open(yaml_fn) as fp:
            for k, v in YAML().load(fp)[config_name].items():
                self.hparams[k] = v

    def flatten(self, d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
