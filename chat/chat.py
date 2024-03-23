import argparse

import torch

from chat.llm_inference import init_OpenAI, make_openai_request, init_local, make_instruct_request, \
    make_completion_request
from chat.utils import ConvHistory, HiddenPrints, parse_json
from chat.retrieval import RAGModule

# silence annoying ragatouille logging
import logging
logging.getLogger("ragatouille").setLevel(logging.CRITICAL)


def make_context_string(rag_results):
    context_string = ""
    for result in rag_results:
        context_string += f"\n\n- {result}"
    return context_string


def make_instruct_query(name, description, conv_history, rag_results):
    context_string = make_context_string(rag_results)
    system = f"You are playing the role of {name}, {description}."
    user = f"You are chatting online. Write your next response in the conversation, based on the following excerpts" \
           f"from {name}'s writing:{context_string}\n\nConversation history:\n\n{conv_history}\n{name}:"
    return system, user


def make_completion_query(name, description, conv_history, rag_results):
    context_string = make_context_string(rag_results)
    prompt_str = f"Character sheet:\n\n{name}: {description}.\n\nExamples of {name}'s writing:{context_string}\n\n" \
                 f"Conversation history:\n\n{conv_history}\n{name}:"
    return prompt_str


def setup(config_path, model_name):
    config = parse_json(config_path)
    rag_module = RAGModule(config, k=3)
    use_openai = model_name.startswith("gpt")
    with HiddenPrints():
        rag_module.search(query="warmup")
    if use_openai:
        client = init_OpenAI()
        instruct = True
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, instruct = init_local(model_name, device)
        client = None
    conv_history = ConvHistory()
    return rag_module, config, use_openai, client, instruct, device, model, tokenizer, conv_history


def make_response(config, query, speaker, conv_history, instruct, rag_module, use_openai, client, model, tokenizer, device, model_name):
    conv_history.add(speaker, query)
    with HiddenPrints():
        results = rag_module.search(query=query)
    if instruct:
        system, user = make_instruct_query(
            config['name'], config['description'], conv_history, results
        )
        prompt = f"{system} {user}"
        if use_openai:
            response = make_openai_request(
                client, system, user, model_name)
        elif instruct:
            response = make_instruct_request(
                model, tokenizer, prompt, device
            )
    else:
        prompt = make_completion_query(
            config['name'], config['description'], conv_history, results
        )
        response = make_completion_request(
            model, tokenizer, prompt, device)
    conv_history.add(config['name'], response)
    return prompt, response


def chat_loop(config_path, show_prompt, model_name):
    rag_module, config, use_openai, client, instruct, device, model, tokenizer, conv_history = setup(
        config_path, model_name
    )
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, response = make_response(
            config, query, "friend", conv_history, instruct, rag_module, use_openai, client, model, tokenizer, device, model_name
        )
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        print(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="configs/zef.json", help="Path to config file")
    parser.add_argument("--show_prompt", "-s",
                        action="store_true", help="Print the system prompt")
    parser.add_argument(
        "--model-name", "-m", default="gpt-3.5-turbo", help="model name (gpt model name or huggingface)")
    parser.add_argument("--device", default="cuda",
                        help="Device to use for local model (cpu or cuda)")
    if not torch.cuda.is_available():
        parser.set_defaults(device="cpu")

    args = parser.parse_args()
    chat_loop(args.config, args.show_prompt, args.model_name)


if __name__ == "__main__":
    main()
