import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from src.bot.chat_controller import ChatController
from src.bot.message import Message
from src.utils.local_logger import LocalLogger


def get_tsv_qa_questions(gt_tsv_file: Path) -> Tuple[List[str], List[str], List[str]]:
    # get all ground truth responses
    with open(gt_tsv_file, "r") as gt_tsv:
        gt_tsv_lines = gt_tsv.readlines()
    # get "response" column from tsv
    gt_tsv_lines = [line.strip() for line in gt_tsv_lines][1:]
    gt_tsv_authors = [line.split("\t")[0] for line in gt_tsv_lines]
    gt_tsv_questions = [line.split("\t")[-2] for line in gt_tsv_lines]
    gt_tsv_responses = [line.split("\t")[-1] for line in gt_tsv_lines]
    return gt_tsv_authors, gt_tsv_questions, gt_tsv_responses


def get_json_qa_questions(gt_json_file: Path) -> Tuple[List[str], List[str]]:
    with open(gt_json_file, "r") as gt_json:
        gt_json_data = json.load(gt_json)
    gt_json_authors = [item["author"] for item in gt_json_data]
    gt_json_questions = [item["question"] for item in gt_json_data]
    gt_json_responses = [item["response"] for item in gt_json_data]
    return gt_json_authors, gt_json_questions, gt_json_responses


def make_output_json(
    authors: List[str],
    questions: List[str],
    responses: List[List[str]],
    ground_truths: List[str],
) -> str:
    lines = [
        json.dumps(
            {
                "author": author,
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
            }
        )
        for author, question, response, ground_truth in zip(
            authors, questions, responses, ground_truths
        )
    ]
    return "\n".join(lines)


def make_output_tsv(
    authors: List[str],
    questions: List[str],
    responses: List[List[str]],
    ground_truths: List[str],
) -> str:
    stringified_responses = [str(response) for response in responses]
    lines = [
        f"{author}\t{question}\t{response}\t{ground_truth}"
        for author, question, response, ground_truth in zip(
            authors, questions, stringified_responses, ground_truths
        )
    ]
    return "\n".join(lines)


def make_answer_file(
    gt_tsv_file: Path,
    config_path: Path,
    out_dir: Path,
    logger: LocalLogger,
    show_prompt: bool,
    sleep_time: int,
    output_format: str,
):
    controller = ChatController(
        config_path,
        logger,
        qa_mode=True,
    )
    if gt_tsv_file.suffix == ".tsv":
        qa_authors, qa_questions, qa_ground_truths = get_tsv_qa_questions(gt_tsv_file)
    elif gt_tsv_file.suffix == ".json":
        qa_authors, qa_questions, qa_ground_truths = get_json_qa_questions(gt_tsv_file)
    else:
        raise ValueError(f"Unsupported file type: {gt_tsv_file.suffix}")
    qa_responses = []
    for author, question in zip(qa_authors, qa_questions):
        message = Message(
            conversation="",
            timestamp=None,
            sender_name=author,
            platform="",
            text_content=question,
            bot_config={},
        )
        controller.update_conv_history(message)
        prompt, responses = controller.make_response(message)
        if show_prompt:
            print("prompt:", prompt)
        print("question:", question)
        print("responses:", responses)
        qa_responses.append(responses)
        time.sleep(sleep_time)
    out_fname = os.path.join(out_dir, f"{config_path.stem}.tsv")
    if output_format == "tsv":
        content = make_output_tsv(
            qa_authors, qa_questions, qa_responses, qa_ground_truths
        )
        out_fname = os.path.join(out_dir, f"{config_path.stem}.tsv")
    elif output_format == "json":
        content = make_output_json(
            qa_authors, qa_questions, qa_responses, qa_ground_truths
        )
        out_fname = os.path.join(out_dir, f"{config_path.stem}.json")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    with open(out_fname, "w") as f:
        f.write(content)
    print(f"Saved responses to {out_fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_tsv_file",
        "-g",
        type=Path,
        default="zef_eval/gt.tsv",
        help="Path to the ground truth tsv file",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=Path,
        help="Path to the config file",
        default="configs/bot/zef_completion.json",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=Path,
        help="Output directory for tsv file",
        default="zef_eval/output",
    )
    parser.add_argument(
        "--console_log_level",
        "-cl",
        type=str,
        help="Console log level",
        default="INFO",
    )
    parser.add_argument(
        "--file_log_level",
        "-fl",
        type=str,
        help="File log level",
        default="INFO",
    )
    parser.add_argument(
        "--show_prompt",
        "-s",
        action="store_true",
        help="Show prompt",
    )
    parser.add_argument(
        "--sleep_time",
        "-t",
        type=int,
        help="Sleep time between requests (in seconds)",
        default=1,
    )
    parser.add_argument(
        "--output_format",
        "-f",
        type=str,
        help="Output format",
        default="json",
        choices=["json", "tsv"],
    )
    parser.add_argument(
        "--log_dir",
        "-l",
        type=Path,
        help="Log directory",
        default="zef_eval/logs",
    )
    args = parser.parse_args()
    logger = LocalLogger(
        args.log_dir, "qa_eval", args.console_log_level, args.file_log_level
    )
    make_answer_file(
        args.gt_tsv_file,
        args.config_path,
        args.out_dir,
        logger,
        args.show_prompt,
        args.sleep_time,
        args.output_format,
    )


if __name__ == "__main__":
    main()
