
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_OpenAI():
    client = OpenAI()
    return client


def init_Anthropic():
    client = Anthropic()
    return client


def make_openai_request(client, system, user, model_name):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return completion.choices[0].message.content


def make_anthropic_request(client, messages, model):
    message = client.messages.create(
        max_tokens=1024,
        messages=messages,
        model=model,
    )
    return message.content[0].text


def init_local(model_name, device, deterministic=False):
    if deterministic:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, num_beams=1, do_sample=False).half().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name)
    return model, tokenizer, "Instruct" in model_name


def make_instruct_request(model, tokenizer, prompt, device):
    prompt = f"[INST]{prompt}[/INST]"
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids)[0]
    return parse_instruct_output(output)


def make_completion_request(model, tokenizer, prompt, device, target_name, chat_user_name, deterministic=False):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=100, do_sample=not deterministic, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids)[0]
    # return output
    return parse_completion_output(output, prompt, target_name, chat_user_name)


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
