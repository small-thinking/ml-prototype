"""
Tokenizer Tool
--------------
Purpose:
    This script provides functionalities to build and use custom tokenizers.
    It supports building a Byte-Pair Encoding (BPE) tokenizer and a SentencePiece tokenizer,
    as well as encoding and decoding text using a pre-trained tokenizer.

Usage:
    1. To build a tokenizer:
        python tokenization_tool.py build -t [bpe|spm] -d [folder_with_text_files] -o [folder_to_save_tokenizer] \
            -v [vocab_size] -m [min_frequency]
    
    2. To encode and decode a text:
        python tokenization_tool.py encode_decode -x [text_to_encode_and_decode] -o [folder_with_trained_tokenizer]

Options:
    -t, --tokenizer_type: Type of the tokenizer to build. Either 'bpe' for BytePair or 'spm' for SentencePiece.
    -d, --text_folder_path: Folder containing the text files to train the tokenizer on.
    -o, --token_folder_path: Folder where the trained tokenizer files will be saved or are already saved.
    -v, --vocab_size: Vocabulary size for the tokenizer. (Default: 5000)
    -m, --min_frequency: Minimum frequency for subwords in the training data. (Default: 2)
    -x, --text: Text to encode and then decode using a pre-trained tokenizer.

Example:
    python tokenization_tool.py build -t bpe -d ./data -o ./tokenizer -v 5000
    python tokenization_tool.py encode_decode -x "Hello, world!" -o ./tokenizer

Author:
    Your Name
"""
import argparse
import os
import random
from typing import List

from tokenizer import BytePairTokenizer, SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tokenizer Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Sub-command for building tokenizer
    build_parser = subparsers.add_parser("build", help="Build a tokenizer")
    build_parser.add_argument(
        "--tokenizer_type",
        "-t",
        type=str,
        default="bpe",
        choices=["bpe", "spm"],
        help="Type of the tokenizer. Either 'bpe' for BytePair or 'spm' for SentencePiece.",
    )
    build_parser.add_argument(
        "--text_folder_path",
        "-d",
        type=str,
        default="./data",
        help="Folder containing the text files to train the tokenizer.",
    )
    build_parser.add_argument(
        "--token_folder_path",
        "-o",
        type=str,
        default="./tokenizer",
        help="Folder to save the trained tokenizer files.",
    )
    build_parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        default=5000,
        help="Vocabulary size for the tokenizer.",
    )
    build_parser.add_argument(
        "--min_frequency",
        "-m",
        type=int,
        default=2,
        help="Minimum frequency for subwords in the training data.",
    )

    # Sub-command for encode-decode
    encode_decode_parser = subparsers.add_parser(
        "encode_decode", help="Encode and then decode a given text"
    )
    encode_decode_parser.add_argument(
        "--text", "-x", type=str, required=True, help="Text to encode and then decode."
    )
    encode_decode_parser.add_argument(
        "--token_folder_path",
        "-o",
        type=str,
        required=True,
        help="Folder where the trained tokenizer files are stored.",
    )

    # Sub-command for merging and sampling .txt files
    merge_parser = subparsers.add_parser(
        "merge_sample", help="Merge and sample txt files"
    )
    merge_parser.add_argument(
        "--folder_path",
        "-d",
        type=str,
        required=True,
        help="Folder containing the .txt files.",
    )
    merge_parser.add_argument(
        "--k",
        "-k",
        type=float,
        required=True,
        help="Percentage of files to sample, should be between 0 and 100.",
    )
    merge_parser.add_argument(
        "--output_filename",
        "-o",
        type=str,
        required=True,
        help="Name of the output file to write merged content to.",
    )

    return parser.parse_args()


def build_tokenizer(args):
    """Build and train a tokenizer based on the provided args."""
    config = {
        "text_folder_path": args.text_folder_path,
        "token_folder_path": args.token_folder_path,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
    }
    if args.tokenizer_type == "bpe":
        tokenizer = BytePairTokenizer(config)
    elif args.tokenizer_type == "spm":
        tokenizer = SentencePieceTokenizer(config)
    else:
        print("Invalid tokenizer type.")
        return

    tokenizer.train()


def encode_decode_text(args):
    """Encode and decode a text using a pre-trained tokenizer."""
    token_folder_path = args.token_folder_path

    spm_model_path = os.path.join(token_folder_path, "spm_model.model")
    bpe_vocab_path = os.path.join(token_folder_path, "vocab.json")
    bpe_merges_path = os.path.join(token_folder_path, "merges.txt")

    if os.path.exists(spm_model_path):
        config = {"token_folder_path": token_folder_path, "tokenizer_type": "spm"}
        tokenizer = SentencePieceTokenizer(config)
    elif os.path.exists(bpe_vocab_path) and os.path.exists(bpe_merges_path):
        config = {"token_folder_path": token_folder_path, "tokenizer_type": "bpe"}
        tokenizer = BytePairTokenizer(config)
    else:
        print("Invalid tokenizer type or missing configuration.")
        return

    text = args.text
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original text: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")


def merge_sample_txt_files(args) -> None:
    """
    Randomly sample k percent of the .txt files from a folder and merge them into one file.

    Args:
    - folder_path (str): Path to the folder containing .txt files.
    - k (float): Percentage of files to sample, should be between 0 and 100.
    - output_filename (str): Name of the output file to write merged content to.

    Returns:
    - None: Writes merged content to output_filename.
    """
    folder_path = args.folder_path
    k = args.k
    output_filename = args.output_filename

    # List all .txt files in the folder
    all_txt_files: List[str] = [
        f for f in os.listdir(folder_path) if f.endswith(".txt")
    ]

    # Calculate the number of files to sample
    num_files_to_sample: int = int(len(all_txt_files) * (k / 100))

    # Randomly sample files
    sampled_files: List[str] = random.sample(all_txt_files, num_files_to_sample)

    # Merge content from sampled files
    merged_content = ""
    total_files = len(sampled_files)
    for idx, file in enumerate(sampled_files):
        with open(os.path.join(folder_path, file), "r") as f:
            merged_content += f.read() + "\n\n"  # Add two newlines between files
        print(f"Processed {idx+1}/{total_files} files.")

    # Write merged content to output file
    with open(output_filename, "w") as f:
        f.write(merged_content)
    print(f"Merged content written to {output_filename}")


def main():
    """Main function."""
    args = parse_args()

    if args.command == "build":
        build_tokenizer(args)
    elif args.command == "encode_decode":
        encode_decode_text(args)
    elif args.command == "merge_sample":
        merge_sample_txt_files(args)
    else:
        print(
            "Invalid command. Use 'build' to build a tokenizer, 'encode_decode' to encode and decode text,"
            " or 'merge_sample' to merge and sample txt files."
        )


if __name__ == "__main__":
    main()
