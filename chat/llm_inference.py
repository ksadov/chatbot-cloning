
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def init_OpenAI():
    client = OpenAI()
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


def init_local(model_name, device, half=False):
    if half:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True).half().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True).to(device)
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


def make_completion_request(model, tokenizer, prompt, device):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids)[0]
    # return output
    return parse_completion_output(output, prompt)


def cleanup_output(output):
    # trim everything after the first instance of "friend", if there is one
    output_trimmed = output.split("friend")[0]
    # trim whitespace in front and back
    output_trimmed = output_trimmed.strip()
    # split by newline, take first line
    output_trimmed = output_trimmed.split("\n")[0]
    # trim before first :, if there is one
    colon = output_trimmed.find(":")
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


def parse_completion_output(output, prompt):
    # trim prompt off of front of output
    output = output[len(prompt):]
    output_trimmed = cleanup_output(output)
    # return output_trimmed
    return output_trimmed


def parse_instruct_output(output):
    # trim everything before [/INST]
    output_trimmed = output.split("[/INST]")[1]
    output_trimmed = cleanup_output(output_trimmed)
    return output_trimmed
