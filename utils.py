import os
import sys
import json


def parse_json(config_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


class ConvHistory:
    def __init__(self, max_length=5):
        self.history = []
        self.max_length = max_length

    def add(self, role, message):
        self.history.append((role, message))
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def __str__(self):
        return "\n".join([f"{role}: {message}" for role, message in self.history])


# RAGatouille prints a lot of stuff to the console, which is annoying
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
