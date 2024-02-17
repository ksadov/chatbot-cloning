This repository allows you to recreate the results of [this blogpost](https://www.ksadov.com/posts/2024-02-16-rag.html).

# Setup
This code has been tested with with Python Python 3.9.13.

1. Activate your virtual env
2. Install Pytorch via the installation instructions given here: https://pytorch.org/get-started/locally/
3. `pip install -r requirements.txt`

You should now be able to run `python chat.py` and see the code build a vector index from the contents of `data/zef.txt`, then drop you into command-line chat loop with the Zef chatbot. By default the script makes calls to the OpenAI API, so you'll need to set the `OPENAI_API_KEY` variable in your shell environment: https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key

# Usage

`chat.py` accepts an `-m` flag that can be either an [OpenAI GPT model](https://platform.openai.com/docs/models/continuous-model-upgrades) or a model on Huggingface, i.e `mistralai/Mistral-7B-v0.1`

You can bring your own data by pasting text chunks seperated by the string `\n-----\n` into a single document, similar to the layout of `data/zef.txt`. Then make a config document of the format `/configs/zef.json` and pass it to `chat.py` via the `-c` flag.

