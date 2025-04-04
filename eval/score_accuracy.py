import os
import argparse
import json

from chat.llm_inference import init_Anthropic, make_anthropic_request


def get_accuracy(client, gt_tsv_questions, gt_tsv_responses, gen_tsv_responses):
    scored = []
    for i in range(len(gt_tsv_responses)):
        question = gt_tsv_questions[i]
        s1 = gt_tsv_responses[i]
        s2 = gen_tsv_responses[i]
        prompt = f"""Answer "True" if the following answers are logically consistent with each other, and "False" otherwise.
        
        Examples:
        Question: What is your Myers-Briggs type?
        Answer 1: Very archetypally ENFP--Ne Fi
        Answer 2: ISFP / 1651 I think
        Identical: False

        Question: Who is your favorite Hindu goddess?
        Answer 1: Kali, or Matangi
        Answer 2: i like kali
        Identical: True

        Question: {question}
        Answer 1: {s1}
        Answer 2: {s2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "Identical:"
            }
        ]
        response = make_anthropic_request(
            client, messages, "claude-3-opus-20240229")
        # trim whitespace and turn into bool
        response = "True" in response
        scored.append(response)
        print("answer 1:", s1)
        print("answer 2:", s2)
        print("score:", response)
    return scored


def get_qa_scores_from_tsv(gt_tsv_file, gen_tsv_file, manual_eval):
    client = init_Anthropic()
    # get all ground truth responses
    with open(gt_tsv_file, 'r') as gt_tsv:
        gt_tsv_lines = gt_tsv.readlines()
    # get "response" column from tsv
    gt_tsv_lines = [line.strip() for line in gt_tsv_lines][1:]
    # split by tab and get last element
    gt_tsv_questions = [line.split('\t')[-2] for line in gt_tsv_lines]
    gt_tsv_responses = [line.split('\t')[-1] for line in gt_tsv_lines]
    gen_tsv_responses_all = []
    with open(gen_tsv_file, 'r') as gen_tsv:
        gen_tsv_lines = gen_tsv.readlines()
    gen_tsv_lines = [line.strip() for line in gen_tsv_lines][1:]
    gen_tsv_responses = [line.split('\t')[-1] for line in gen_tsv_lines]
    gen_tsv_responses_all.append(gen_tsv_responses)
    if manual_eval:
        qa_scores = [False] * len(gt_tsv_responses)
    else:
        qa_scores = get_accuracy(
            client, gt_tsv_questions, gt_tsv_responses, gen_tsv_responses)
    return qa_scores, gt_tsv_responses, gen_tsv_responses_all


def save_accuracies(gen_tsv_file, score_dict, gt_tsv_responses, gen_tsv_responses_all, save_dir):
    # save as json of response pairs and qa scores, seperate file per gen file
    out_fname = "accuracy_" + \
        os.path.basename(gen_tsv_file)[:-4] + ".json"
    json_list = []
    qa_scores = score_dict
    qa_score_file = os.path.join(
        save_dir, out_fname)
    with open(qa_score_file, 'w') as f:
        for i in range(len(gt_tsv_responses)):
            s1 = gt_tsv_responses[i]
            s2 = gen_tsv_responses_all[0][i]
            score = qa_scores[i]
            json_list.append(
                {'response_1': s1, 'response_2': s2, 'qa_score': str(score)})
        json.dump(json_list, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_tsv_file', '-gt',
                        default="eval/zef/gt.tsv", type=str)
    parser.add_argument('--gen_tsv_file', '-gen', type=str)
    parser.add_argument('--save_dir', '-s',
                        default="eval/zef/output", type=str)
    parser.add_argument("--manual_eval", "-m", action="store_true")
    args = parser.parse_args()

    # if save_dir does not exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    score_dict, gt_tsv_responses, gen_tsv_responses_all = get_qa_scores_from_tsv(
        args.gt_tsv_file, args.gen_tsv_file, args.manual_eval
    )
    save_accuracies(args.gen_tsv_file, score_dict, gt_tsv_responses,
                    gen_tsv_responses_all, args.save_dir)


if __name__ == "__main__":
    main()
