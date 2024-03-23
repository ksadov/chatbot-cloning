import argparse
import discord
import os

from chat.chat import setup, make_response
from chat.utils import parse_json

intents = discord.Intents.default()
intents.message_content = True


class ChatBot(discord.Client):
    def __init__(self, config_path, discord_config, model_name):
        super().__init__(intents=intents)
        self.rag_module, self.config, self.use_openai, self.openai_client, self.instruct, self.device, self.model, \
            self.tokenizer, self.conv_history = setup(config_path, model_name)
        self.discord_config = discord_config

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    async def on_message(self, message):
        if message.author == self.user:
            return
        elif message.channel.name in self.discord_config['channels']:
            if message.content == self.discord_config['clear_command']:
                self.conv_history.clear()
                await message.channel.send("[Conversation history cleared]")
            else:
                prompt, response = make_response(
                    self.config, message.content, message.author.name, self.conv_history, self.instruct, self.rag_module, self.use_openai,
                    self.openai_client, self.model, self.tokenizer, self.device, self.config[
                        'name'],
                    message.channel.name
                )
                print("------------------")
                print(prompt)
                print("------------------")
                await message.channel.send(response)


def main():
    # get discord token from environment
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str,
                        help='Path to the config file', default='configs/zef.json')
    parser.add_argument('--discord_config_path', '-d', type=str,
                        help='Path to the discord config file', default='discord/configs/zef.json')
    parser.add_argument('--model_name', '-m', type=str,
                        help='Name of the model to use', default='mistralai/Mistral-7B-v0.1')
    args = parser.parse_args()
    discord_config = parse_json(args.discord_config_path)
    chatbot = ChatBot(args.config_path,
                      discord_config,
                      args.model_name)
    chatbot.run(discord_config['token'])


if __name__ == '__main__':
    main()
