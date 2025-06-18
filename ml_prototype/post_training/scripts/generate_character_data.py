from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import tqdm
import argparse
import datasets
from huggingface_hub import login
from typing import Literal


def load_prompt_template(category: str, stage: Literal["topic_gen", "question_gen", "data_gen"]) -> dict[str, str]:
    """
    Load the prompt template for the given stage.
    The prompt templates for each category is in a folder in prompts folder.
    And in the folder there should be 3 files:
    - topic_gen_prompt.txt
    - question_gen_prompt.txt
    - data_gen_prompt.txt
    The prompt template for the given stage is the content of the file.
    """
    prompt_template_path = os.path.join(os.path.dirname(__file__), "prompts", category, stage + "_prompt.txt")
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(f"Prompt template file not found: {prompt_template_path}")
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def generate_topics(category: str, use_deepseek: bool = False):
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY") if not use_deepseek else os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key) if not use_deepseek else OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")

    # Generate 100 big topics
    try:
        topic_gen_prompt = load_prompt_template(category=category, stage="topic_gen")
        print(f"Loaded topic gen prompt: {topic_gen_prompt}")
    except FileNotFoundError as e:
        print(f"Error loading topic gen prompt: {e}")
        exit(1)

    print("Generating topics...")
    response = client.chat.completions.create(
        model="gpt-4o-mini" if not use_deepseek else "deepseek-chat",
        messages=[
            {"role": "system", "content": topic_gen_prompt},
        ],
        response_format={
            "type": "json_object",
        }
    )

    # Parse the response as a list of dicts
    print(f"Response: {response.choices[0].message.content}")
    topics = json.loads(response.choices[0].message.content)["data"]
    topic_list = [{"name": topic["name"], "description": topic["description"]} for topic in topics]
    print("Generated topics")

    # Save the topics to a jsonl file
    # Create folder if not exists
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "topics.jsonl")
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "a") as f:
        for topic in tqdm.tqdm(topic_list):
            f.write(json.dumps(topic) + "\n")

    print("Saved topics")


def generate_questions(
    category: str,
    use_deepseek: bool = False,
):
    """
    Read the topics from the jsonl file, and generate 100 questions for each topic.
    The questions should be in the style of Trump.
    The questions should be in English.
    The questions should be in the format of:
    {"topic": "<concise English prompt>", "theme": "{{theme}}"}
    The questions should be saved to a jsonl file.
    """

    topics_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "topics.jsonl")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")

    prompt_template = load_prompt_template(category=category, stage="question_gen")
    print(f"Loaded question gen prompt: {prompt_template}")

    api_key = os.getenv("OPENAI_API_KEY") if not use_deepseek else os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key) if not use_deepseek else OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")

    # Load the topics from the jsonl file
    topics_jsonl_path = os.path.join(os.path.dirname(__file__), topics_jsonl_path)
    with open(topics_jsonl_path, "r") as f:
        themes = [json.loads(line) for line in f]
    print(f"Loaded {len(themes)} themes from {topics_jsonl_path}")

    # Generate the questions for each topic
    with open(output_jsonl_path, "a") as f:
        for theme in tqdm.tqdm(themes):
            prompt = prompt_template.format(theme=theme["name"])
            response = client.chat.completions.create(
                model="gpt-4o-mini" if not use_deepseek else "deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            try:
                questions = json.loads(response.choices[0].message.content)["data"]
                for question in questions:
                    try:
                        if "topic" in question:
                            f.write(json.dumps({"topic": question["topic"], "theme": theme["name"]}) + "\n")
                        else:
                            print(f"Error: {question} is not a valid question. Expected format: {{'topic': '...', 'theme': '...'}}")
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Problematic content: {question}")
            except json.JSONDecodeError as e:
                tqdm.tqdm.write(f"Error decoding JSON for theme {theme['name']}: {e}")
                tqdm.tqdm.write(f"Problematic content: {response.choices[0].message.content}")
            except Exception as e:
                tqdm.tqdm.write(f"An unexpected error occurred for theme {theme['name']}: {e}")

    print(f"Generated questions and saved to {output_jsonl_path}")


def generate_data(
    category: str,
    use_deepseek: bool = False,
    batch_size: int = 10,
):
    topics_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")

    api_key = os.getenv("OPENAI_API_KEY") if not use_deepseek else os.getenv("DEEPSEEK_API_KEY")

    # Deepseek API
    client = OpenAI(api_key=api_key) if not use_deepseek else OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")

    prompt_template = load_prompt_template(category=category, stage="data_gen")
    print(f"Loaded data gen prompt: {prompt_template}")

    # Load the questions/conversation starters from the jsonl file
    with open(topics_jsonl_path, "r") as f:
        topics = [json.loads(line) for line in f]
    print(f"Loaded {len(topics)} topics from {topics_jsonl_path}")

    # Generate the data for each question, batch by 5 questions
    # Create file if not exists
    data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)
    print(f"Write to data jsonl path: {data_jsonl_path}")
    if not os.path.exists(data_jsonl_path):
        with open(data_jsonl_path, "w") as f:
            pass

    with open(data_jsonl_path, "a", encoding="utf-8") as f:
        # Wrap the outer loop with tqdm for a progress bar
        # desc will show a description next to the progress bar
        for i in tqdm.tqdm(range(0, len(topics), batch_size), desc="Generating & Saving Data"):
            # Construct the topic string from the list of topic for the current batch
            topics_str = ", ".join([topic["topic"] for topic in topics[i:i+batch_size]])
            prompt = prompt_template.format(topics=topics_str)

            # Generate the data for each topics batch
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini" if not use_deepseek else "deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                # Parse the response as a list of dicts
                data = json.loads(response.choices[0].message.content)["data"]
                # Save the data to a jsonl file
                for record in data:
                    # Use ensure_ascii=False to write actual Chinese characters for human readability
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                tqdm.tqdm.write(f"Error decoding JSON for batch {i}: {e}")
                tqdm.tqdm.write(f"Problematic content: {response.choices[0].message.content}")
            except Exception as e:
                tqdm.tqdm.write(f"An unexpected error occurred for batch {i}: {e}")


def convert_data_to_sft_data(
    category: str,
    request_key_suffix: str = "topic",
    response_key_suffix: str = "contrarian",
):
    """
    Convert the data to a format that can be used for SFT.
    The format of the data is:
    {
        "en_prompt": "...",
        "cn_prompt": "...",
        "en_chosen": "...",
        "cn_chosen": "...",
        "en_rejected": "...",
        "cn_rejected": "..."
    }
    For each such record, we will generate 2 records, one for en, one for cn.
    {"messages": [{"role": "user", "content": "original en prompt"}, {"role": "assistant", "content": "original en response"}]}
    {"messages": [{"role": "user", "content": "original cn prompt"}, {"role": "assistant", "content": "original cn response"}]}
    """
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, f"{response_key_suffix}_sft_data.jsonl")

    print(f"Converting data from {data_jsonl_path} to {output_jsonl_path}")
    with open(data_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(data):
            try:
                f.write(
                    json.dumps({
                        "messages":
                        [
                            {"role": "user", "content": record["en_" + request_key_suffix]},
                            {"role": "assistant", "content": record["en_" + response_key_suffix]}
                        ]}, ensure_ascii=False) + "\n"
                )
                f.write(
                    json.dumps({
                        "messages":
                        [
                            {"role": "user", "content": record["cn_" + request_key_suffix]},
                            {"role": "assistant", "content": record["cn_" + response_key_suffix]}
                        ]}, ensure_ascii=False) + "\n"
                )
            except Exception as e:
                tqdm.tqdm.write(f"An unexpected error occurred for record {record}: {e}")

    print(f"Converted data from {data_jsonl_path} to {output_jsonl_path}")


def convert_data_to_dpo_data(
    category: str,
    request_key_suffix: str = "topic",
    chosen_key_suffix: str = "contrarian",
):
    """
    Convert the data to a format that can be used for DPO.
    The format of the data is:
    {
        "en_prompt": "...",
        "cn_prompt": "...",
        "en_chosen": "...",
        "cn_chosen": "...",
        "en_rejected1": "...",
        "cn_rejected1": "...",
        "en_rejected2": "...",
        "cn_rejected2": "..."
    }
    For each such record, we will generate multiple jsonl records, one for each rejected key suffix.
    Each record will have the format:
    {
        "chosen": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "chosen"}],
        "rejected": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "rejected"}]
    }
    """
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, f"{chosen_key_suffix}_dpo_data.jsonl")

    with open(data_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if not data:
        raise ValueError("No data found in the input file")

    # Infer rejected key suffixes from the first record
    first_record = data[0]
    rejected_key_suffixes = []
    for key in first_record.keys():
        if key.startswith("en_") and key != f"en_{request_key_suffix}" and key != f"en_{chosen_key_suffix}":
            rejected_key_suffixes.append(key[3:])  # Remove "en_" prefix

    if not rejected_key_suffixes:
        raise ValueError(f"No rejected key suffixes found in the data. Make sure there are keys other than {request_key_suffix} and {chosen_key_suffix}")

    print(f"Found rejected key suffixes: {rejected_key_suffixes}")

    with open(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(data):
            try:
                # Generate pairs for English
                for rejected_suffix in rejected_key_suffixes:
                    f.write(
                        json.dumps({
                            "chosen": [
                                {"role": "user", "content": record["en_" + request_key_suffix]},
                                {"role": "assistant", "content": record["en_" + chosen_key_suffix]}
                            ],
                            "rejected": [
                                {"role": "user", "content": record["en_" + request_key_suffix]},
                                {"role": "assistant", "content": record["en_" + rejected_suffix]}
                            ]
                        }, ensure_ascii=False) + "\n"
                    )

                # Generate pairs for Chinese
                for rejected_suffix in rejected_key_suffixes:
                    f.write(
                        json.dumps({
                            "chosen": [
                                {"role": "user", "content": record["cn_" + request_key_suffix]},
                                {"role": "assistant", "content": record["cn_" + chosen_key_suffix]}
                            ],
                            "rejected": [
                                {"role": "user", "content": record["cn_" + request_key_suffix]},
                                {"role": "assistant", "content": record["cn_" + rejected_suffix]}
                            ]
                        }, ensure_ascii=False) + "\n"
                    )
            except Exception:
                continue


def upload_data_to_hf(
    category: str,
    dataset_type: Literal["sft", "dpo"],
    prefix: str = "",
):
    """
    Upload the data to Hugging Face.
    """
    load_dotenv(override=True)
    dataset_name = f"{category}_{prefix}_{dataset_type}_data"
    hf_token = os.getenv("HF_TOKEN")
    print(f"Uploading data to Hugging Face: {dataset_name}")
    login(token=hf_token)
    print("Logged in to Hugging Face")
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, f"{prefix}_{dataset_type}_data.jsonl")
    print(f"Loading data from {data_jsonl_path}")
    dataset = datasets.load_dataset("json", data_files=data_jsonl_path)
    dataset.push_to_hub("/".join(["tech-tao", dataset_name]))


def arg_parser():
    """
    Parse the arguments.
    Example command:
    python generate_character_data.py gen-topic --use_deepseek --category gang-jing
    python generate_character_data.py gen-question --use_deepseek --category gang-jing
    python generate_character_data.py gen-data --use_deepseek --batch_size 10 --category gang-jing
    python generate_character_data.py convert-sft --category gang-jing --request_key_suffix topic --response_key_suffix contrarian
    python generate_character_data.py convert-dpo --category gang-jing --request_key_suffix topic --chosen_key_suffix contrarian
    python generate_character_data.py upload-hf --category gang-jing --dataset_type sft --prefix contrarian
    """
    parser = argparse.ArgumentParser(description="A CLI tool to generate character data for post-training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create generate topic subparser
    generate_topic_parser = subparsers.add_parser("gen-topic", help="Generate topics")
    generate_topic_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_topic_parser.add_argument("--use_deepseek", action="store_true", help="Use deepseek")
    
    # Create question subparser
    generate_question_parser = subparsers.add_parser("gen-question", help="Generate questions")
    generate_question_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_question_parser.add_argument("--use_deepseek", action="store_true", help="Use deepseek")

    # Create generate data subparser
    generate_data_parser = subparsers.add_parser("gen-data", help="Generate data")
    generate_data_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_data_parser.add_argument("--use_deepseek", action="store_true", help="Use deepseek")
    generate_data_parser.add_argument("--batch_size", type=int, default=10, help="Batch size")

    # Create convert data to sft data subparser
    convert_data_to_sft_parser = subparsers.add_parser("convert-sft", help="Convert data to SFT data")
    convert_data_to_sft_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    convert_data_to_sft_parser.add_argument("--request_key_suffix", type=str, default="topic", help="Suffix of the request key")
    convert_data_to_sft_parser.add_argument("--response_key_suffix", type=str, default="contrarian", help="Suffix of the response key")

    # Create convert data to dpo data subparser
    convert_data_to_dpo_parser = subparsers.add_parser("convert-dpo", help="Convert data to DPO data")
    convert_data_to_dpo_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    convert_data_to_dpo_parser.add_argument("--request_key_suffix", type=str, default="topic", help="Suffix of the request key")
    convert_data_to_dpo_parser.add_argument("--chosen_key_suffix", type=str, default="contrarian", help="Suffix of the chosen key")

    # Create upload data to hf subparser
    upload_data_to_hf_parser = subparsers.add_parser("upload-hf", help="Upload data to Hugging Face")
    upload_data_to_hf_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    upload_data_to_hf_parser.add_argument("--dataset_type", type=str, default="sft", help="Type of the dataset")
    upload_data_to_hf_parser.add_argument("--prefix", type=str, default="", help="Prefix of the dataset")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        exit(1)
    return args


if __name__ == "__main__":
    args = arg_parser()
    if args.command == "gen-topic":
        generate_topics(args.category, args.use_deepseek)
    elif args.command == "gen-question":
        generate_questions(args.category, args.use_deepseek)
    elif args.command == "gen-data":
        generate_data(args.category, args.use_deepseek, args.batch_size)
    elif args.command == "convert-sft":
        convert_data_to_sft_data(args.category, args.request_key_suffix, args.response_key_suffix)
    elif args.command == "convert-dpo":
        convert_data_to_dpo_data(args.category, args.request_key_suffix, args.chosen_key_suffix)
    elif args.command == "upload-hf":
        upload_data_to_hf(args.category, args.dataset_type, args.prefix)
