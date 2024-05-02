from chat.chat import setup, make_response
import os
import argparse


def get_qa_questions(gt_tsv_file):
    # get all ground truth responses
    with open(gt_tsv_file, 'r') as gt_tsv:
        gt_tsv_lines = gt_tsv.readlines()
    # get "response" column from tsv
    gt_tsv_lines = [line.strip() for line in gt_tsv_lines][1:]
    gt_tsv_authors = [line.split('\t')[0] for line in gt_tsv_lines]
    gt_tsv_questions = [line.split('\t')[-2] for line in gt_tsv_lines]
    return gt_tsv_authors, gt_tsv_questions


def make_answer_tsv(gt_tsv_file, config_path, model_name, k, out_dir):
    rag_module, config, use_openai, openai_client, instruct, device, model, \
        tokenizer, conv_history = setup(
            config_path, model_name, k=k)
    qa_authors, qa_questions = get_qa_questions(gt_tsv_file)
    qa_responses = []
    for author, question in zip(qa_authors, qa_questions):
        _, response = make_response(
            config, question, author, conv_history, instruct, rag_module, use_openai,
            openai_client, model, tokenizer, device, model_name, None
        )
        print("question:", question)
        print("response:", response)
        qa_responses.append(response)
    response_tsv_lines = []
    for author, question, response in zip(qa_authors, qa_questions, qa_responses):
        response_tsv_lines.append(f"{author}\t{question}\t{response}")
    out_fname_suffix = os.path.basename(config_path).split('.')[
        0] + "-" + model_name.replace('/', '_')
    out_fname = os.path.join(out_dir, f"qa_{out_fname_suffix}.tsv")
    with open(out_fname, 'w') as f:
        f.write("author\tquestion\tanswer\n")
        for line in response_tsv_lines:
            f.write(line + "\n")
    print(f"Saved responses to {out_fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_tsv_file', '-g', type=str,
                        default='eval/zef/gt.tsv',
                        help='Path to the ground truth tsv file')
    parser.add_argument('--config_path', '-c', type=str,
                        help='Path to the config file', default='configs/zef.json')
    parser.add_argument('--model_name', '-m', type=str,
                        help='Name of the model to use', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--k', type=int,
                        help='Number of documents to retrieve', default=3)
    parser.add_argument('--out_dir', '-o', type=str,
                        help='Output directory for tsv file', default='eval/zef/gen')
    args = parser.parse_args()
    make_answer_tsv(args.gt_tsv_file, args.config_path,
                    args.model_name, args.k, args.out_dir)


if __name__ == '__main__':
    main()
