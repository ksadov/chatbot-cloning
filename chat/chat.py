import argparse
import datetime
import torch

from chat.llm_inference import init_OpenAI, make_openai_request, init_local, make_instruct_request, \
    make_completion_request
from chat.utils import HiddenPrints, parse_json
from chat.conversation import ConvHistory, Message
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


def setup(config_path, model_name, k):
    config = parse_json(config_path)
    rag_module = RAGModule(config, k)
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
    conv_history = ConvHistory(config["include_timestamp"])
    return rag_module, config, use_openai, client, instruct, device, model, tokenizer, conv_history


def make_response(config, query, speaker, conv_history, instruct, rag_module, use_openai, client, model, tokenizer, device, model_name, conversation_name):
    query_timestamp = datetime.datetime.now()
    conv_history.add(
        Message(conversation_name, query_timestamp, speaker, query))
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
    response_timestamp = datetime.datetime.now()
    conv_history.add(
        Message(conversation_name, response_timestamp, config['name'], response))
    return prompt, response


def chat_loop(config_path, show_prompt, model_name, k):
    rag_module, config, use_openai, client, instruct, device, model, tokenizer, conv_history = setup(
        config_path, model_name, k
    )
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, response = make_response(
            config, query, "friend", conv_history, instruct, rag_module, use_openai, client, model, tokenizer, device, model_name, None
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
    parser.add_argument("--k", default=3, type=int,)
    if not torch.cuda.is_available():
        parser.set_defaults(device="cpu")

    args = parser.parse_args()
    chat_loop(args.config, args.show_prompt, args.model_name, args.k)


if __name__ == "__main__":
    main()
