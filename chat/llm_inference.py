from transformers import pipeline, AutoTokenizer
import json

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


def make_completion_query(name, chat_user_name, description, conv_history, rag_results, include_timestamp):
    context_string = make_context_string(rag_results)
    if include_timestamp:
        timestamp_str = f"[{dt.now().strftime('%Y-%m-%d %H:%M')}] "
    else:
        timestamp_str = ""
    prompt_str = f"Character sheet:\n\n{name}: {description}.\n\nExamples of {name}'s writing:{context_string}\n\n" \
                 f"Conversation history:\n\n{conv_history}\n{timestamp_str}{name}:"
    return prompt_str

class LLM:
    def __init__(self, config, device):
        self.config = config
        self.model_name = self.config["model"]
        self.device = device
        self.instruct = self.config["instruct"]

    def chat_step(self, name, chat_user_name, description, conv_history, rag_results, include_timestamp):
        if self.instruct:
            system, user = make_instruct_query(name, description, conv_history, rag_results)
            prompt = f"[INST]{system}[/INST]{user}"
            response = self.make_instruct_request(prompt)
        else:
            prompt = make_completion_query(name, chat_user_name, description, conv_history, rag_results, include_timestamp)
            response = self.make_completion_request(prompt, name, "friend", deterministic=False)
        return prompt, response


    def make_instruct_request(self, prompt):
        pass

    def make_completion_request(self, prompt):
        pass

class LocalLLM(LLM):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.pipe = pipeline("text-generation", model=self.model_name, device=device, trust_remote_code=True, torch_dtype='float16')
        self.instruct = config["instruct"]


    def make_instruct_request(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        output = self.pipe(messages)
        return parse_instruct_output(output)


    def make_completion_request(self, prompt, target_name, chat_user_name, deterministic=False):
        output = self.pipe(prompt)[0]['generated_text']
        return parse_completion_output(output, prompt, target_name, chat_user_name)

class RemoteLLM(LLM):
    def __init__(self, confi, device):
        super().__init__(config, device)
        self.api_base = self.config["api_base"]
        self.api_key = self.config["api_key"]
        self.model = self.config["model"]

    def make_instruct_request(self, system, user):
        # use chat completion api
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        return response.json()["choices"][0]["message"]["content"]

    def make_completion_request(self, prompt):
        # use completion api
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": prompt},
        )
        return response.json()["choices"][0]["text"]


def setup_llm(config_path, device):
    config = json.load(open(config_path))
    if config["api_base"] is not None:
        return RemoteLLM(config, device)
    else:
        return LocalLLM(config, device)


def cleanup_output(output, target_name, chat_user_name):
    # trim everything after the first instance of responder name, if there is one
    output_trimmed = output.split(chat_user_name)[0]
    # trim whitespace in front and back
    output_trimmed = output_trimmed.strip()
    # split by newline, take first line
    output_trimmed = output_trimmed.split("\n")[0]
    # trim before first :, if there is one
    colon = output_trimmed.find(f"{target_name}:") + len(target_name) + 1
    if colon != -1:
        output_trimmed = output_trimmed[colon+1:]
    # trim anything after the last period, unless there are no periods
    periods = [i for i, c in enumerate(output_trimmed) if c == "."]
    if len(periods) > 0:
        output_trimmed = output_trimmed[:periods[-1]+1]
    # get rid of <s> and </s>
    output_trimmed = output_trimmed.replace("<s>", "")
    output_trimmed = output_trimmed.replace("</s>", "")
    # trim whitespace again
    output_trimmed = output_trimmed.strip()
    return output_trimmed


def parse_completion_output(output, prompt, target_name, chat_user_name):
    # trim prompt off of front of output
    output = output[len(prompt):]
    output_trimmed = cleanup_output(output, target_name, chat_user_name)
    # return output_trimmed
    return output_trimmed


def parse_instruct_output(output):
    # trim everything before [/INST]
    output_trimmed = output.split("[/INST]")[1]
    output_trimmed = cleanup_output(output_trimmed)
    return output_trimmed
