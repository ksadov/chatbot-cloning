import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

from src.bot.chat_controller import ChatController


def get_qa_questions(gt_tsv_file: Path) -> Tuple[List[str], List[str]]:
    # get all ground truth responses
    with open(gt_tsv_file, "r") as gt_tsv:
        gt_tsv_lines = gt_tsv.readlines()
    # get "response" column from tsv
    gt_tsv_lines = [line.strip() for line in gt_tsv_lines][1:]
    gt_tsv_authors = [line.split("\t")[0] for line in gt_tsv_lines]
    gt_tsv_questions = [line.split("\t")[-2] for line in gt_tsv_lines]
    return gt_tsv_authors, gt_tsv_questions


def make_answer_tsv(
    gt_tsv_file: Path,
    config_path: Path,
    out_dir: Path,
    console_log_level: str,
    file_log_level: str,
    show_prompt: bool,
    sleep_time: int,
):
    controller = ChatController(
        config_path,
        log_dir=out_dir / "logs",
        console_log_level=console_log_level,
        file_log_level=file_log_level,
        qa_mode=True,
    )
    qa_authors, qa_questions = get_qa_questions(gt_tsv_file)
    qa_responses = []
    for author, question in zip(qa_authors, qa_questions):
        prompt, responses = controller.make_response(question, author, "conversation")
        if show_prompt:
            print("prompt:", prompt)
        print("question:", question)
        response_str = str(responses)
        print("responses:", response_str)
        qa_responses.append(response_str)
        time.sleep(sleep_time)
    response_tsv_lines = []
    for author, question, response in zip(qa_authors, qa_questions, qa_responses):
        response_tsv_lines.append(f"{author}\t{question}\t{response}")
    out_fname = os.path.join(out_dir, f"{config_path.stem}.tsv")
    with open(out_fname, "w") as f:
        f.write("author\tquestion\tanswer\n")
        for line in response_tsv_lines:
            f.write(line + "\n")
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
        "-l",
        type=str,
        help="Console log level",
        default="INFO",
    )
    parser.add_argument(
        "--file_log_level",
        "-f",
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
    args = parser.parse_args()
    make_answer_tsv(
        args.gt_tsv_file,
        args.config_path,
        args.out_dir,
        args.console_log_level,
        args.file_log_level,
        args.show_prompt,
        args.sleep_time,
    )


if __name__ == "__main__":
    main()
