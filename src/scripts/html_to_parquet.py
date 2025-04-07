import argparse
import os

import bs4
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chat.conversation import Message


def format_discord_date(date):
    # Discord date format: 10/22/2023 6:06 PM
    return pd.to_datetime(date, format="%m/%d/%Y %I:%M\u202f%p")


def process_individual_discord_message(html_file, message, soup):
    conversation = html_file.split("/")[-1].replace(".html", "")
    user_id_span = message.find("span", class_="chatlog__author")
    if user_id_span:
        user_id = user_id_span["title"]
    else:
        user_id = None
    message_time_span = message.find("span", class_="chatlog__timestamp")
    if message_time_span:
        message_time = message_time_span.find("a").text
        message_time = format_discord_date(message_time)
    else:
        message_time = None
    text_span = message.find("span", class_="chatlog__markdown-preserve")
    if text_span:
        content = text_span.text
    else:
        content = "[Attachment]"
    return Message(conversation, message_time, user_id, content)


def process_all_messages(html_file, output_dir):
    # if output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parquet_file = os.path.join(
        output_dir, html_file.split("/")[-1].replace(".html", ".parquet")
    )
    # if file already exists, delete it
    if os.path.exists(parquet_file):
        os.remove(parquet_file)
    message_class = "chatlog__message-container"
    messages = []
    with open(html_file, "r") as file:
        soup = bs4.BeautifulSoup(file, "html.parser")
        for message in soup.find_all("div", class_=message_class):
            try:
                processed = process_individual_discord_message(html_file, message, soup)
                if processed.user_id is None:
                    if len(messages) > 0:
                        processed.user_id = messages[-1].user_id
                    else:
                        print("No user_id found for message:\n", message)
                        continue
                if processed.timestamp is None:
                    processed.timestamp = messages[-1].timestamp
                messages.append(processed)
            except Exception as e:
                print("error parsing message:\n", message)
                print(e)
                print("Resuming parsing...")

    arrow_messages = [message.to_arrow() for message in messages]
    table = pa.Table.from_batches(arrow_messages)
    pq.write_table(table, parquet_file)


def process_all_html_files(html_dir, output_dir):
    for filename in os.listdir(html_dir):
        if filename.endswith(".html"):
            print(f"Processing {filename}")
            process_all_messages(os.path.join(html_dir, filename), output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Discord HTML chat logs to Parquet"
    )
    parser.add_argument(
        "--html_dir",
        type=str,
        help="Directory containing HTML files to process",
        default="data/discord_raw",
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        help="Directory to write Parquet files to",
        default="data/discord_parquet",
    )
    args = parser.parse_args()
    process_all_html_files(args.html_dir, args.parquet_dir)


if __name__ == "__main__":
    main()
