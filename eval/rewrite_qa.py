from chat.llm_inference import init_Anthropic, make_anthropic_request
import argparse


def get_rewrites(questions, answers, style_description):
    client = init_Anthropic()
    rewritten = []
    for question, answer in zip(questions, answers):
        if question == "x":
            rewritten.append(None)
        else:
            prompt = f"Rewrite the following answer to the question \"{question}\" {style_description}: {answer}"
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            response = make_anthropic_request(
                client, messages, "claude-3-haiku-20240307")
            rewritten.append(response)
            print("rewritten:", response)
    return rewritten


def get_rewrites_from_tsv(tsv_file, style_description):
    with open(tsv_file, 'r') as tsv:
        tsv_lines = tsv.readlines()
    tsv_lines = [line.strip() for line in tsv_lines][1:]
    authors = [line.split('\t')[0] for line in tsv_lines]
    questions = [line.split('\t')[1] for line in tsv_lines]
    answers = [line.split('\t')[2] for line in tsv_lines]
    rewrites = get_rewrites(questions, answers, style_description)
    # create a new tsv with the same authors and questions, but the rewrites as the answers
    new_tsv_lines = []
    for author, question, rewrite in zip(authors, questions, rewrites):
        if rewrite is None:
            new_tsv_lines.append(f"{author}\t{question}\tx")
        else:
            new_tsv_lines.append(f"{author}\t{question}\t{rewrite}")
    new_tsv_file = tsv_file.replace('.tsv', '_rewrites.tsv')
    with open(new_tsv_file, 'w') as new_tsv:
        for line in new_tsv_lines:
            new_tsv.write(f"{line}\n")
    return rewrites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_file', type=str)
    parser.add_argument('--style_description', type=str,
                        default="in an erudite style, in two sentences or less")
    args = parser.parse_args()
    get_rewrites_from_tsv(args.tsv_file, args.style_description)


if __name__ == '__main__':
    main()
