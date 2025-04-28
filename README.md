This repository allows you to create a chatbot based on someone's writing that you can interact with over Discord or through the command line. You can read blog posts about earlier iterations of the codebase [here](https://www.ksadov.com/series/Chatbot%20Cloning.html).

# Setup
## Environment
This code has been tested with with Python 3.9.13.

1. Activate your virtual env
2. Install Pytorch via the installation instructions given here: https://pytorch.org/get-started/locally/
3. `pip install -r requirements.txt`

## Quickstart
1. Set your OpenAI key in:
  - `configs/embedding/text-embedding-3-small.json`
  - `configs/llm/gpt-4o-mini.json`
2. `python -m src.scripts.serve_retrieval --config configs/retrieval/zef_demo_gt.json --port 5000`
3. In another terminal window: `python -m src.scripts.serve_retrieval --config configs/retrieval/zef_demo_conv_history.json --port 5001`
4. In yet another window: `python -m src.scripts.chat --bot_config_path configs/bot/zef_demo.json`

Step 4 will drop you into a command-line chat loop with a bot based on the contents of `data/zef.txt`. Read on to learn how to change your bot's source data, prompt, LLM backend and more.

# Customizing your bot
## Retrieval
### Background
The bot can be configured to retrieve from two vector stores:
- A store of the cloning target's ground-truth writing, for example, the Facebook status updates in `data/zef.txt`.
- A store of previous chatbot conversation, which can be updated over time in order to allow the chatbot to learn new things.

Either or both of these stores are optional for the bot's operation, but you probably want to use them in order to give the bot some prior context to go off of.

The bot interacts with vector stores as servers: see `src/bot/rag_module.py`. I've included an implementation of a vector store server in `src.retrieval`: what follows are instructions for running my implementation.

### Embedding model config
In order to produce vectors to put in these stores, we need an embedding model. My implementation allows the user to specify a local embedding model with Huggingface (see `configs/embedding/bge-large-en-v1.5.json`) or a hosted embedding model (see `configs/embedding/text-embedding-3-small.json`).

Once you've created your embedding model config you'll set its path as the value of `embedding_config_path` in your retrieval config.

### Retrieval config
`configs/retrieval` contains examples of configs for the two types of vector store:
- `configs/retrieval/zef_demo_gt.json` is a ground-truth store for the contents of the document `data/zef.txt`. It does not permit updates.
- `configs/retrieval/zef_demo_conv_history.json` is a chatbot conversation store. It starts out empty and get updated with new messages over time.

If you want to use your own data, you can either:
- create a .txt file in the same format as `data/zef.txt`, with individual samples separated by the string `\n-----\n`
- create a parquet document (or folder of parquet documents) where each entry has the fields `text` (specifying the text to embed) and an optional dictionary field `meta` (specifying metadata associated with the entry)

Once you have a retrieval config that you're satisfied with, you can serve it using `python -m src.scripts.serve_retrieval --config configs/retrieval/my_store.json`.

## Chat
### Prompt template
You can specify the format to use when presenting information to your model with a Jinja template. `configs/prompt_templates` contains two examples of such templates:
- `zef_completion.j2` is designed to be used with base models like Mixtral-8x7B-v0.1 which try to continue the output of whatever input they got.
- `zef_instruct.j2` is designed to be used with instruction models like OpenAI's GPT or Anthropic's Claude.

### LLM config
You can use any inference endpoint which implements the [OpenAI Chat Completions spec](https://platform.openai.com/docs/api-reference/chat) (which includes many non-OpenAI providers, like [Together AI](https://docs.together.ai/reference/chat-completions-1)) or [Anthropic's messages API](https://docs.anthropic.com/en/api/messages) (though tool use is unsupported). See `configs/llm` for examples.
### Config
`configs/bot/` contains examples of a config designed for base model inference and a config designed for instruct inference.

Once you have a bot config that you're satisfied with, you can chat with in from the command line with `python -m src.scripts.chat --bot_config_path configs/bot/my_config.json`.

## Discord
To chat with your bot on Discord, you'll need to [make a Discord bot account and acquire a token](https://www.writebots.com/discord-bot-token/). Then you'll need to create a Discord bot config: see `configs/discord` for an example. You'll need to specify the following fields:
- `channels`: a list of channel names that the bot can talk in in any server that it's invited to. The bot will also respond to any DMs that you send it.
- `clear_command`: tying this string in a channel that the bot can access will clear its recent conversational memory, which is useful if it's become stuck in a loop.
- `token`: the token for your bot's account

Once you have a config, run your bot with `python -m src.scripts.run_discord_bot --bot_config_path configs/bot/my_config.json --discord_config_path configs/bot/my_discord_config.json`.

## Evaluate
When testing out bots, you may want to run different configurations on some standard set of questions to compare outputs. `src/scripts/qa_eval` will let you do this given as input either a:
- .json file containing a list of entries with the fields `author`, `question` and `response`, where `author` is a string representing the question's author and `response` is a ground-truth answer that you'd consider "correct".
- .tsv file containing columns representing author, question and response

It will then output a json or tsv file (depending on command-line args) that allows you to compare generated answer to the specified ground-truth response.

## Message database
You can specify a database to save messages that the bots sends and receives via the `-db` argument to `src.scripts.chat` and `src.scripts.run_discord_bot`. This codebase only supports storing to a local SQLite database for now, see `configs/database/sqlite_example.json` for an example.
