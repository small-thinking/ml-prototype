from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import tqdm
import argparse
import datasets
from huggingface_hub import login


def generate_topics(use_deepseek: bool = False, output_jsonl_path: str = "./data/topics.jsonl"):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # Generate 100 big topics
    topic_gen_prompt = """
    You are a helpful assistant to generate synthetic data for Trump character post-training.
    We would first generate 100 diverse topics (less about politics). These topics will be used to generate prompts.
    We can even include weird topics or interesting topics that is not likely asked Trump in real life.

    Please for each topic generate a name, with a short description.
    The output should be a jsonl compatible format, each with the following fields:
    - "name": the name of the topic
    - "description": the description of the topic

    For example, output rows each is a json object in a line:
    [
        {{"name": "Science", "description": "A topic about science"}},
        {{"name": "Political", "description": "A topic about politics"}}
    ]

    """

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
    print(type(response.choices[0].message.content))

    # Parse the response as a list of dicts
    print(response.choices[0].message.content)
    topics = json.loads(response.choices[0].message.content)
    topic_list = [{"name": topic["name"], "description": topic["description"]} for topic in topics]
    print("Generated topics")

    # Save the topics to a jsonl file
    output_jsonl_path = os.path.join(os.path.dirname(__file__), output_jsonl_path)
    with open(output_jsonl_path, "a") as f:
        for topic in tqdm.tqdm(topic_list):
            f.write(json.dumps(topic) + "\n")

    print("Saved topics")


def generate_questions():
    pass


def generate_data(
        use_deepseek: bool = False,
        question_jsonl_path: str = "./data/questions.jsonl",
        data_jsonl_path: str = "./data/trump_conv_data.jsonl",
        batch_size: int = 10,
):
    api_key = os.getenv("OPENAI_API_KEY") if not use_deepseek else os.getenv("DEEPSEEK_API_KEY")

    # Deepseek API
    client = OpenAI(api_key=api_key) if not use_deepseek else OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")

    prompt_template = """
    You are to respond in the distinctive speaking style of Donald J. Trump.

    Style of Trump includes:
    - Short, punchy sentences.
    - Frequent repetition of key phrases.
    - Confident, self-praising tone.
    - Exaggerated claims (e.g., tremendous, disaster, fake).
    - Blame others if needed.
    - Reasoning can be shallow, but must sound assertive and concrete.
    - Can demonstrate his wealthy lifestyle.
    - Each response should be 3-5 sentences with around 50-100 words.

    You will be given a list of questions in English. Your task is to:
    1. Translate the question into Chinese.
    2. Answer the question twice:
    - Once in Trump's style (in both English and Chinese).
    - Once in a neutral, academic style (in both English and Chinese).

    Return a single-line JSON object with:
    {{
    "data": [
        {{
        "en_prompt": "...",
        "cn_prompt": "...",
        "en_chosen": "...",    # Trump-style English response
        "cn_chosen": "...",    # Trump-style Chinese response
        "en_rejected": "...",  # Neutral English response
        "cn_rejected": "..."   # Neutral Chinese response
        }},
        ...
    ]
    }}

    ### Example 1
    Questions: ["What do you think about artificial intelligence?", "How do you feel about climate change?"]
    Output:
    {{
    "data": [
        {{"en_prompt": "What do you think about artificial intelligence?", "cn_prompt": "你怎么看人工智能？", "en_chosen": "AI? It's big. It's powerful. We’re gonna use it and dominate. I knew it since I were young. Nobody other than Elon does it better than me.", "cn_chosen": "人工智能？太厉害了。我们要用它来称霸。我很年轻的时候就知道这玩意儿了。除了Elon没人比我更会用。", "en_rejected": "Artificial intelligence is a transformative technology that presents both opportunities and risks. We must regulate it responsibly.", "cn_rejected": "人工智能是一项具有变革性的技术，既带来机遇，也伴随风险。我们需要负责任地进行监管。"}},
        {{"en_prompt": "How do you feel about climate change?", "cn_prompt": "你怎么看气候变化？", "en_chosen": "Climate change? Total hoax. China loves it. It’s killing our jobs. We need strong energy, American energy!", "cn_chosen": "气候变化？彻头彻尾的骗局。中国高兴得很。但它毁了我们的工作岗位。我们需要强大的能源，美国的能源！", "en_rejected": "Climate change is a serious global issue that requires international cooperation and long-term environmental policy.", "cn_rejected": "气候变化是一个全球性问题，需要国际合作和长期的环保政策。"}}
    ]
    }}

    ### Now you try
    Question: [{questions}]
    Output:
    """

    # Load the questions/conversation starters from the jsonl file
    question_jsonl_path = os.path.join(os.path.dirname(__file__), question_jsonl_path)
    with open(question_jsonl_path, "r") as f:
        questions = [json.loads(line) for line in f]

    # Generate the data for each question, batch by 5 questions
    # Create file if not exists
    data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)
    if not os.path.exists(data_jsonl_path):
        with open(data_jsonl_path, "w") as f:
            pass

    with open(data_jsonl_path, "a", encoding="utf-8") as f:
        # Wrap the outer loop with tqdm for a progress bar
        # desc will show a description next to the progress bar
        for i in tqdm.tqdm(range(1588, len(questions), batch_size), desc="Generating & Saving Data"):
            # Construct the questions string from the list of questions for the current batch
            questions_str = ", ".join([question["question"] for question in questions[i:i+batch_size]])
            prompt = prompt_template.format(questions=questions_str)

            # Generate the data for each question batch
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
    data_jsonl_path: str = "./data/trump_conv_data.jsonl",
    output_jsonl_path: str = "./data/trump_conv_sft_data.jsonl"
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
    data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)
    output_jsonl_path = os.path.join(os.path.dirname(__file__), output_jsonl_path)

    print(f"Converting data from {data_jsonl_path} to {output_jsonl_path}")
    with open(data_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(data):
            f.write(
                json.dumps({
                    "messages":
                    [
                        {"role": "user", "content": record["en_prompt"]},
                        {"role": "assistant", "response": record["en_chosen"]}
                    ]}, ensure_ascii=False) + "\n"
            )
            f.write(
                json.dumps({
                    "messages":
                    [
                        {"role": "user", "content": record["cn_prompt"]},
                        {"role": "assistant", "content": record["cn_chosen"]}
                    ]}, ensure_ascii=False) + "\n"
            )
    print(f"Converted data from {data_jsonl_path} to {output_jsonl_path}")


def convert_data_to_dpo_data(
    data_jsonl_path: str = "./data/trump_conv_data.jsonl",
    output_jsonl_path: str = "./data/trump_conv_dpo_data.jsonl"
):
    """
    Convert the data to a format that can be used for DPO.
    The format of the data is:
    {
        "en_prompt": "...",
        "cn_prompt": "...",
        "en_chosen": "...",
        "cn_chosen": "...",
        "en_rejected": "...",
        "cn_rejected": "..."
    }
    For each such record, we will generate 2 jsonl records, one for en, one for cn.
    {"chosen": [{"role": "user", "content": "original en prompt"}, {"role": "assistant", "content": "original en response"}], "rejected": [{"role": "user", "content": "original en prompt"}, {"role": "assistant", "content": "original en response"}]}
    {"chosen": [{"role": "user", "content": "original cn prompt"}, {"role": "assistant", "content": "original cn response"}], "rejected": [{"role": "user", "content": "original cn prompt"}, {"role": "assistant", "content": "original cn response"}]}
    """
    data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)
    output_jsonl_path = os.path.join(os.path.dirname(__file__), output_jsonl_path)

    with open(data_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(data):
            f.write(
                json.dumps({"chosen": [{"role": "user", "content": record["en_prompt"]}, {"role": "assistant", "content": record["en_chosen"]}], "rejected": [{"role": "user", "content": record["en_prompt"]}, {"role": "assistant", "content": record["en_rejected"]}]}, ensure_ascii=False) + "\n"
            )
            f.write(
                json.dumps({"chosen": [{"role": "user", "content": record["cn_prompt"]}, {"role": "assistant", "content": record["cn_chosen"]}], "rejected": [{"role": "user", "content": record["cn_prompt"]}, {"role": "assistant", "content": record["cn_rejected"]}]}, ensure_ascii=False) + "\n"
            )


def arg_parser():
    """
    Parse the arguments.
    Example command:
    python generate_character_data.py gen-topic --use_deepseek --output_jsonl_path ./data/topics.jsonl
    python generate_character_data.py gen-data --use_deepseek --question_jsonl_path ./data/questions.jsonl --data_jsonl_path ./data/trump_conv_data.jsonl --batch_size 10
    python generate_character_data.py convert-sft --data_jsonl_path ./data/trump_conv_data.jsonl --output_jsonl_path ./data/trump_conv_sft_data.jsonl
    python generate_character_data.py convert-dpo --data_jsonl_path ./data/trump_conv_data.jsonl --output_jsonl_path ./data/trump_conv_dpo_data.jsonl
    python generate_character_data.py upload-hf --data_jsonl_path ./data/trump_conv_data.jsonl --dataset_name trump_conv_data
    """
    parser = argparse.ArgumentParser(description="A CLI tool to generate character data for post-training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create generate topic subparser
    generate_topic_parser = subparsers.add_parser("gen-topic", help="Generate topics")
    generate_topic_parser.add_argument("--use_deepseek", action="store_false", help="Use deepseek")
    generate_topic_parser.add_argument("--output_jsonl_path", type=str, default="./data/topics.jsonl", help="Path to save the topics")

    # Create generate data subparser
    generate_data_parser = subparsers.add_parser("gen-data", help="Generate data")
    generate_data_parser.add_argument("--use_deepseek", action="store_false", help="Use deepseek")
    generate_data_parser.add_argument("--question_jsonl_path", type=str, default="./data/questions.jsonl", help="Path to the questions jsonl file")
    generate_data_parser.add_argument("--data_jsonl_path", type=str, default="./data/trump_conv_data.jsonl", help="Path to save the data")
    generate_data_parser.add_argument("--batch_size", type=int, default=10, help="Batch size")

    # Create convert data to sft data subparser
    convert_data_to_sft_parser = subparsers.add_parser("convert-sft", help="Convert data to SFT data")
    convert_data_to_sft_parser.add_argument("--data_jsonl_path", type=str, default="./data/trump_conv_data.jsonl", help="Path to the data jsonl file")
    convert_data_to_sft_parser.add_argument("--output_jsonl_path", type=str, default="./data/trump_conv_sft_data.jsonl", help="Path to save the SFT data")

    # Create convert data to dpo data subparser
    convert_data_to_dpo_parser = subparsers.add_parser("convert-dpo", help="Convert data to DPO data")
    convert_data_to_dpo_parser.add_argument("--data_jsonl_path", type=str, default="./data/trump_conv_data.jsonl", help="Path to the data jsonl file")
    convert_data_to_dpo_parser.add_argument("--output_jsonl_path", type=str, default="./data/trump_conv_dpo_data.jsonl", help="Path to save the DPO data")

    # Create upload data to hf subparser
    upload_data_to_hf_parser = subparsers.add_parser("upload-hf", help="Upload data to Hugging Face")
    upload_data_to_hf_parser.add_argument("--data_jsonl_path", type=str, default="./data/trump_conv_data.jsonl", help="Path to the data jsonl file")
    upload_data_to_hf_parser.add_argument("--dataset_name", type=str, default="trump_conv_data", help="Name of the dataset")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        exit(1)
    return args


def upload_data_to_hf(
    data_jsonl_path: str,
    dataset_name: str,
):
    """
    Upload the data to Hugging Face.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)
    dataset = datasets.load_dataset("json", data_files=data_jsonl_path)
    dataset.push_to_hub("/".join(["john02171574", dataset_name]))


if __name__ == "__main__":
    args = arg_parser()
    if args.command == "gen-topic":
        generate_topics(args.use_deepseek, args.output_jsonl_path)
    elif args.command == "gen-data":
        generate_data(args.use_deepseek, args.question_jsonl_path, args.data_jsonl_path, args.batch_size)
    elif args.command == "convert-sft":
        convert_data_to_sft_data(args.data_jsonl_path, args.output_jsonl_path)
    elif args.command == "convert-dpo":
        convert_data_to_dpo_data(args.data_jsonl_path, args.output_jsonl_path)
    elif args.command == "upload-hf":
        upload_data_to_hf(args.data_jsonl_path, args.dataset_name)
