import os
import sys
import json


def parse_json(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

# RAGatouille prints a lot of stuff to the console, which is annoying


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
