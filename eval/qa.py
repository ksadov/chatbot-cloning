from sentence_transformers import SentenceTransformer
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import json
import torch


def init_embed_model(model_name, fp16):
    # convert model to fp16 if specified
    if fp16:
        embed_model = SentenceTransformer(
            model_name, device='cpu', trust_remote_code=True)
        embed_model = embed_model.eval().half()
        # move model to cuda
        embed_model = embed_model.to('cuda')
    else:
        embed_model = SentenceTransformer(
            model_name, device='cuda', trust_remote_code=True).eval()
    return embed_model


def get_embed_dist(embed_model, s1, s2):
    s1_embed = embed_model.encode(s1)
    s2_embed = embed_model.encode(s2)
    # take the cosine similarity of the two embeddings
    scores = (s1_embed @ s2_embed.T) / \
        (np.linalg.norm(s1_embed) * np.linalg.norm(s2_embed))
    return scores


def get_qa_scores(response_tsv_1, response_tsv_2, embed_model):
    qa_scores = []
    for i in range(len(response_tsv_1)):
        s1 = response_tsv_1[i]
        s2 = response_tsv_2[i]
        scores = get_embed_dist(embed_model, s1, s2)
        qa_scores.append(scores)
    return qa_scores


def get_qa_scores_from_tsv(gt_tsv_file, gen_tsv_dir, embed_model_name, fp16):
    embed_model = init_embed_model(embed_model_name, fp16)
    # get all ground truth responses
    with open(gt_tsv_file, 'r') as gt_tsv:
        gt_tsv_lines = gt_tsv.readlines()
    # get "response" column from tsv
    gt_tsv_lines = [line.strip() for line in gt_tsv_lines][1:]
    # split by tab and get last element
    gt_tsv_responses = [line.split('\t')[-1] for line in gt_tsv_lines]
    # get all generated responses
    gen_tsv_files = [f for f in os.listdir(gen_tsv_dir) if f.endswith('.tsv')]
    gen_tsv_files.sort()
    gen_tsv_responses_all = []
    qa_score_dict = {}
    for gen_tsv_file in gen_tsv_files:
        gen_tsv_file = os.path.join(gen_tsv_dir, gen_tsv_file)
        gen_name = os.path.basename(gen_tsv_file).split('.')[0]
        with open(gen_tsv_file, 'r') as gen_tsv:
            gen_tsv_lines = gen_tsv.readlines()
        gen_tsv_lines = [line.strip() for line in gen_tsv_lines][1:]
        gen_tsv_responses = [line.split('\t')[-1] for line in gen_tsv_lines]
        gen_tsv_responses_all.append(gen_tsv_responses)
        qa_scores = get_qa_scores(
            gt_tsv_responses, gen_tsv_responses, embed_model)
        qa_score_dict[gen_name] = qa_scores
    return qa_score_dict, gt_tsv_responses, gen_tsv_responses_all


def save_qa_score_dict(response_tsv_1, response_tsvs, qa_score_dict, embed_model_name, save_dir):
    # save as json of response pairs and qa scores, seperate file per gen file
    for key, response_tsv in zip(qa_score_dict.keys(), response_tsvs):
        json_list = []
        qa_scores = qa_score_dict[key]
        model_fname = embed_model_name.replace('/', '_')
        qa_score_file = os.path.join(
            save_dir, f'qa_scores_{model_fname}_{key}.json')
        with open(qa_score_file, 'w') as f:
            for i in range(len(response_tsv_1)):
                s1 = response_tsv_1[i]
                s2 = response_tsv[i]
                score = qa_scores[i]
                json_list.append(
                    {'response_1': s1, 'response_2': s2, 'qa_score': str(score)})
            json.dump(json_list, f)


def plot_qa_scores(qa_score_dict, model_name, save_dir):
    for gen_name, qa_scores in qa_score_dict.items():
        plt.hist(qa_scores, bins=20, alpha=0.5, label=gen_name)
    plt.legend(loc='upper right')
    plt.xlabel('QA Scores')
    plt.ylabel('Frequency')
    plt.title('QA Scores Distribution')
    fname_model = model_name.replace('/', '_')
    plt.savefig(os.path.join(save_dir, f'qa_scores_{fname_model}.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_tsv_file', '-gt',
                        default="eval/zef/gt.tsv", type=str)
    parser.add_argument('--gen_tsv_dir', '-gen',
                        default="eval/zef/gen", type=str)
    parser.add_argument('--embed_model_name', '-e', type=str, required=True)
    parser.add_argument('--save_dir', '-s',
                        default="eval/zef/output", type=str)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    # if save_dir does not exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    qa_score_dict, gt_tsv_responses, gen_tsv_responses_all = get_qa_scores_from_tsv(
        args.gt_tsv_file, args.gen_tsv_dir, args.embed_model_name, args.fp16)
    save_qa_score_dict(gt_tsv_responses, gen_tsv_responses_all,
                       qa_score_dict, args.embed_model_name, args.save_dir)
    plot_qa_scores(qa_score_dict, args.embed_model_name, args.save_dir)


if __name__ == '__main__':
    main()
