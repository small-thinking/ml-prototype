from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import tqdm
import argparse
import datasets
from huggingface_hub import login
from typing import Literal
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


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


def extract_json_from_response(content: str) -> dict | None:
    """
    Extracts a JSON object from a string, handling markdown code blocks.
    """
    try:
        if "```" in content:
            # Assumes the JSON is inside a markdown block
            # Handles ```json ... ``` or ``` ... ```
            json_str = content.split("```")[1]
            if json_str.lower().strip().startswith("json"):
                json_str = json_str.strip()[4:].strip()
            return json.loads(json_str)
        else:
            # Assumes the content is a plain JSON string
            return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        # Fallback to find json between first { and last }
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            return None


def get_client(provider: Literal["openai", "deepseek", "anthropic"]) -> OpenAI | anthropic.Anthropic:
    """
    Get the API client for the given provider.
    """
    load_dotenv(override=True)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_topics(category: str, use_deepseek: bool = False):
    load_dotenv(override=True)

    client = get_client("deepseek" if use_deepseek else "openai")

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

    client = get_client("deepseek" if use_deepseek else "openai")

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
                            print(
                                f"Error: {question} is not a valid question."
                                f" Expected format: {{'topic': '...', 'theme': '...'}}"
                            )
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Problematic content: {question}")
            except json.JSONDecodeError as e:
                tqdm.tqdm.write(f"Error decoding JSON for theme {theme['name']}: {e}")
                tqdm.tqdm.write(f"Problematic content: {response.choices[0].message.content}")
            except Exception as e:
                tqdm.tqdm.write(f"An unexpected error occurred for theme {theme['name']}: {e}")

    print(f"Generated questions and saved to {output_jsonl_path}")


def generate_data_openai(
    category: str,
    use_deepseek: bool = False,
    batch_size: int = 10,
):
    topics_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")

    client = get_client("deepseek" if use_deepseek else "openai")

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


def generate_data_claude_batch(category: str, model: str = "claude-sonnet-4-0"):
    """
    Generate data using Claude's batch API.
    This function creates and submits a batch request.
    """
    questions_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")
    batch_id_path = os.path.join(os.path.dirname(__file__), "./data", category, "claude_batch_id.txt")

    client = get_client("anthropic")
    prompt_template = load_prompt_template(category=category, stage="data_gen")

    with open(questions_jsonl_path, "r") as f:
        lines = f.readlines()
    questions = [json.loads(line) for line in lines]
    prompts = [prompt_template.format(topic=q["topic"]) for q in questions]
    requests_list = [
        Request(
            custom_id=f"request_{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        for i, prompt in enumerate(prompts)
    ]

    print(f"Submitting {len(requests_list)} requests to Claude's batch API...")
    batch_job = client.messages.batches.create(requests=requests_list)
    print(f"Batch job created with ID: {batch_job.id}")

    with open(batch_id_path, "w") as f:
        f.write(batch_job.id)
    print(f"Batch ID saved to {batch_id_path}")


def retrieve_claude_batch_results(category: str):
    """
    Retrieve the results of a Claude batch job and format them.
    The output will be a JSONL file where each line is a JSON object
    containing the original topic and the generated responses.
    """
    batch_id_path = os.path.join(os.path.dirname(__file__), "./data", category, "claude_batch_id.txt")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")
    questions_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")

    if not os.path.exists(batch_id_path):
        print(f"Batch ID file not found at {batch_id_path}")
        return

    with open(batch_id_path, "r") as f:
        batch_id = f.read().strip()

    with open(questions_jsonl_path, "r") as f:
        questions = [json.loads(line) for line in f]

    client: anthropic.Anthropic = get_client("anthropic")

    print(f"Checking status for batch job: {batch_id}")
    while True:
        batch_job = client.messages.batches.retrieve(batch_id)
        if batch_job.processing_status == "ended":
            print("Batch job completed.")
            break
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} - Batch {batch_id} is still processing...")
        time.sleep(10)

    print("Retrieving and processing results...")
    with open(output_jsonl_path, "a", encoding="utf-8") as f_out:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                try:
                    request_index = int(custom_id.split("_")[1])
                    original_question = questions[request_index]

                    response_message = result.result.message
                    response_content = response_message.content[0].text
                    llm_record = extract_json_from_response(response_content)

                    if not llm_record:
                        print(f"Warning: Could not extract JSON from response for {custom_id}, content: {response_content}")
                        continue

                    # Construct the final record
                    final_record = {"en_topic": original_question["topic"], "cn_topic": ""}

                    if "cn_topic" in llm_record:
                        final_record["cn_topic"] = llm_record["cn_topic"]

                    # Dynamically add response keys based on the category
                    for key_suffix in ["normal", category]:
                        en_key = f"en_{key_suffix}"
                        cn_key = f"cn_{key_suffix}"
                        if en_key in llm_record:
                            final_record[en_key] = llm_record[en_key]
                        if cn_key in llm_record:
                            final_record[cn_key] = llm_record[cn_key]

                    f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")

                except (KeyError, IndexError, AttributeError) as e:
                    response_content = ""
                    if result.result.type == "succeeded" and result.result.message.content:
                        response_content = result.result.message.content[0].text
                    print(f"Error processing result for {custom_id}: {e}, content: '{response_content}'")
            elif result.result.type == "errored":
                print(f"Request {custom_id} failed: {result.result.error}")
            else:
                print(f"Unknown result type for {custom_id}: {result.result.type}")

    print(f"Results saved to {output_jsonl_path}")


def generate_and_retrieve_claude_data(category: str, model: str = "claude-4.0-sonnet"):
    """
    Submits a Claude batch job and waits for the results.
    """
    print(f"Starting synchronous Claude batch job for category: {category}")
    generate_data_claude_batch(category, model)
    print("Batch request submitted. Now waiting for results...")
    retrieve_claude_batch_results(category)
    print("Claude batch job finished and results retrieved.")


def convert_data_to_sft_data(
    category: str,
    request_key_suffix: str = "topic",
    response_key_suffix: str = "contrarian",
):
    """
    Convert the data to a format that can be used for SFT.
    The format of the data is:
    {
        "messages": [
            {"role": "user", "content": "original en prompt"},
            {"role": "assistant", "content": "original en response"}
        ]
    }
    {
        "messages": [
            {"role": "user", "content": "original cn prompt"},
            {"role": "assistant", "content": "original cn response"}
        ]
    }
    """
    data_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "conv_data.jsonl")
    output_jsonl_path = os.path.join(
        os.path.dirname(__file__), "./data", category, f"{response_key_suffix}_sft_data.jsonl"
    )

    print(f"Converting data from {data_jsonl_path} to {output_jsonl_path}")
    with open(data_jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    with open(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(data):
            try:
                f.write(
                    json.dumps({
                        "messages": [
                            {"role": "user", "content": record["en_" + request_key_suffix]},
                            {"role": "assistant", "content": record["en_" + response_key_suffix]}
                        ]}, ensure_ascii=False) + "\n"
                )
                f.write(
                    json.dumps({
                        "messages": [
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
    output_jsonl_path = os.path.join(
        os.path.dirname(__file__), "./data", category, f"{chosen_key_suffix}_dpo_data.jsonl"
    )

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
        raise ValueError(
            f"No rejected key suffixes found in the data."
            f"Make sure there are keys other than {request_key_suffix} and {chosen_key_suffix}"
        )

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


def generate_questions_claude_batch(category: str, model: str = "claude-sonnet-4-0"):
    """
    Generate questions using Claude's batch API.
    This function creates and submits a batch request for question generation.
    """
    topics_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "topics.jsonl")
    batch_id_path = os.path.join(os.path.dirname(__file__), "./data", category, "claude_question_batch_id.txt")

    client = get_client("anthropic")
    prompt_template = load_prompt_template(category=category, stage="question_gen")

    with open(topics_jsonl_path, "r") as f:
        topics = [json.loads(line) for line in f]

    requests_list = [
        Request(
            custom_id=f"request_{i}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt_template.format(theme=topic["name"])}],
            ),
        )
        for i, topic in enumerate(topics)
    ]

    print(f"Submitting {len(requests_list)} question generation requests to Claude's batch API...")
    batch_job = client.messages.batches.create(requests=requests_list)
    print(f"Batch job for question generation created with ID: {batch_job.id}")

    with open(batch_id_path, "w") as f:
        f.write(batch_job.id)
    print(f"Batch ID saved to {batch_id_path}")


def retrieve_claude_question_batch_results(category: str):
    """
    Retrieve the results of a Claude batch job for question generation.
    """
    batch_id_path = os.path.join(os.path.dirname(__file__), "./data", category, "claude_question_batch_id.txt")
    output_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "questions.jsonl")
    topics_jsonl_path = os.path.join(os.path.dirname(__file__), "./data", category, "topics.jsonl")

    if not os.path.exists(batch_id_path):
        print(f"Batch ID file not found at {batch_id_path}")
        return

    with open(batch_id_path, "r") as f:
        batch_id = f.read().strip()

    with open(topics_jsonl_path, "r") as f:
        topics = [json.loads(line) for line in f]

    client: anthropic.Anthropic = get_client("anthropic")

    print(f"Checking status for question generation batch job: {batch_id}")
    while True:
        batch_job = client.messages.batches.retrieve(batch_id)
        if batch_job.processing_status == "ended":
            print("Batch job completed.")
            break
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')} - Batch {batch_id} is still processing...")
        time.sleep(10)

    print("Retrieving and processing question results...")

    with open(output_jsonl_path, "a", encoding="utf-8") as f_out:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                try:
                    request_index = int(custom_id.split("_")[1])
                    original_topic = topics[request_index]

                    response_message = result.result.message
                    response_content = response_message.content[0].text

                    response_data = extract_json_from_response(response_content)

                    if not response_data or "data" not in response_data:
                        print(f"Warning: Could not extract JSON or 'data' key not in response for {custom_id}, content: {response_content}, parsed data: {response_data}")
                        continue

                    questions = response_data["data"]
                    for question in questions:
                        if "topic" in question:
                            f_out.write(json.dumps({"topic": question["topic"], "theme": original_topic["name"]}) + "\n")
                        else:
                            print(f"Warning: 'topic' key not found in question for {custom_id}: {question}")

                except (KeyError, IndexError, AttributeError) as e:
                    response_content = ""
                    if result.result.type == "succeeded" and result.result.message.content:
                        response_content = result.result.message.content[0].text
                    print(f"Error processing result for {custom_id}: {e}, content: '{response_content}'")
            elif result.result.type == "errored":
                print(f"Request {custom_id} failed: {result.result.error}")
            else:
                print(f"Unknown result type for {custom_id}: {result.result.type}")

    print(f"Question results saved to {output_jsonl_path}")


def generate_and_retrieve_claude_questions(category: str, model: str = "claude-sonnet-4-0"):
    """
    Submits a Claude batch job for question generation and waits for the results.
    """
    print(f"Starting synchronous Claude batch job for question generation for category: {category}")
    generate_questions_claude_batch(category, model)
    print("Batch request for questions submitted. Now waiting for results...")
    retrieve_claude_question_batch_results(category)
    print("Claude question batch job finished and results retrieved.")


def arg_parser():
    """
    Parse the arguments.
    Example command:
    python generate_character_data.py gen-topic --use_deepseek --category gang-jing
    python generate_character_data.py gen-question --use_deepseek --category gang-jing
    python generate_character_data.py gen-question-claude --category gang-jing
    python generate_character_data.py retrieve-question-claude-results --category gang-jing
    python generate_character_data.py gen-question-claude-sync --category gang-jing
    python generate_character_data.py gen-data-openai --use_deepseek --batch_size 10 --category gang-jing
    python generate_character_data.py gen-data-claude --category gang-jing
    python generate_character_data.py retrieve-claude-results --category gang-jing
    python generate_character_data.py gen-claude-sync --category gang-jing
    python generate_character_data.py convert-sft --category yizhipian --request_key_suffix topic --response_key_suffix yizhipian
    python generate_character_data.py convert-dpo --category yizhipian --request_key_suffix topic --chosen_key_suffix yizhipian
    python generate_character_data.py upload-hf --category yizhipian --dataset_type dpo --prefix yizhipian
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

    # Create question claude subparser
    generate_question_claude_parser = subparsers.add_parser(
        "gen-question-claude", help="Generate questions with Claude Batch API"
    )
    generate_question_claude_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_question_claude_parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-0", help="Claude model to use"
    )

    # Create retrieve question claude results subparser
    retrieve_question_claude_parser = subparsers.add_parser(
        "retrieve-question-claude-results", help="Retrieve Claude batch question results"
    )
    retrieve_question_claude_parser.add_argument("--category", type=str, default="trump", help="Category of the data")

    # Create generate question claude sync subparser
    generate_question_claude_sync_parser = subparsers.add_parser(
        "gen-question-claude-sync", help="Generate questions with Claude Batch API and wait for results."
    )
    generate_question_claude_sync_parser.add_argument(
        "--category", type=str, default="trump", help="Category of the data"
    )
    generate_question_claude_sync_parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-0", help="Claude model to use"
    )

    # Create generate data openai subparser
    generate_data_openai_parser = subparsers.add_parser("gen-data-openai", help="Generate data with OpenAI API")
    generate_data_openai_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_data_openai_parser.add_argument("--use_deepseek", action="store_true", help="Use deepseek")
    generate_data_openai_parser.add_argument("--batch_size", type=int, default=10, help="Batch size")

    # Create generate data claude subparser
    generate_data_claude_parser = subparsers.add_parser("gen-data-claude", help="Generate data with Claude Batch API")
    generate_data_claude_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_data_claude_parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-0", help="Claude model to use"
    )

    # Create retrieve claude results subparser
    retrieve_claude_parser = subparsers.add_parser("retrieve-claude-results", help="Retrieve Claude batch results")
    retrieve_claude_parser.add_argument("--category", type=str, default="trump", help="Category of the data")

    # Create generate data claude sync subparser
    generate_data_claude_sync_parser = subparsers.add_parser(
        "gen-claude-sync", help="Generate data with Claude Batch API and wait for results."
    )
    generate_data_claude_sync_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    generate_data_claude_sync_parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-0", help="Claude model to use"
    )

    # Create convert data to sft data subparser
    convert_data_to_sft_parser = subparsers.add_parser("convert-sft", help="Convert data to SFT data")
    convert_data_to_sft_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    convert_data_to_sft_parser.add_argument(
        "--request_key_suffix", type=str, default="topic", help="Suffix of the request key"
    )
    convert_data_to_sft_parser.add_argument(
        "--response_key_suffix", type=str, default="contrarian", help="Suffix of the response key"
    )

    # Create convert data to dpo data subparser
    convert_data_to_dpo_parser = subparsers.add_parser("convert-dpo", help="Convert data to DPO data")
    convert_data_to_dpo_parser.add_argument("--category", type=str, default="trump", help="Category of the data")
    convert_data_to_dpo_parser.add_argument(
        "--request_key_suffix", type=str, default="topic", help="Suffix of the request key"
    )
    convert_data_to_dpo_parser.add_argument(
        "--chosen_key_suffix", type=str, default="contrarian", help="Suffix of the chosen key"
    )

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
    elif args.command == "gen-question-claude":
        generate_questions_claude_batch(args.category, args.model)
    elif args.command == "retrieve-question-claude-results":
        retrieve_claude_question_batch_results(args.category)
    elif args.command == "gen-question-claude-sync":
        generate_and_retrieve_claude_questions(args.category, args.model)
    elif args.command == "gen-data-openai":
        generate_data_openai(args.category, args.use_deepseek, args.batch_size)
    elif args.command == "gen-data-claude":
        generate_data_claude_batch(args.category, args.model)
    elif args.command == "retrieve-claude-results":
        retrieve_claude_batch_results(args.category)
    elif args.command == "gen-claude-sync":
        generate_and_retrieve_claude_data(args.category, args.model)
    elif args.command == "convert-sft":
        convert_data_to_sft_data(args.category, args.request_key_suffix, args.response_key_suffix)
    elif args.command == "convert-dpo":
        convert_data_to_dpo_data(args.category, args.request_key_suffix, args.chosen_key_suffix)
    elif args.command == "upload-hf":
        upload_data_to_hf(args.category, args.dataset_type, args.prefix)
