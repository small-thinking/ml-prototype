# This file includes the code to preprocess the data for the multi-modality item embedding learning.
import json
import os
from tqdm import tqdm
from PIL import Image
import torch


def get_text_index(text_file_folder: str, show_summary: bool = False) -> dict[str, object]:
    """Load the text json files from the folder and return the dictionary of the text content.
    Args:
        text_file_folder: The folder containing the text json files.
        show_summary: Whether to print the summary of the text files.
    Returns:
        The dictionary of the text content. Key: item_id, Value: {"item_name": str, "image_ids": List[str]}
    """
    text_dict = {}
    text_file_folder = os.path.expanduser(text_file_folder)
    json_files = [file for file in os.listdir(text_file_folder) if file.endswith(".json")]
    for file in tqdm(json_files, desc="Processing files"):
        with open(os.path.join(text_file_folder, file), "r") as f:
            for line in f:
                json_line = json.loads(line)
                if "item_id" in json_line and "item_name" in json_line and "main_image_id" in json_line:
                    item_id = json_line["item_id"]
                    item_name = json_line["item_name"][0]["value"]
                    main_image_id = json_line["main_image_id"]
                    text_dict[item_id] = {"item_name": item_name, "image_ids": [main_image_id]}
                    if "other_image_id" in json_line:
                        text_dict[item_id]["image_ids"].extend(json_line["other_image_id"])
    if show_summary:
        print(f"Found {len(text_dict)} text files in the folder.")
        print(f"Example text file: {list(text_dict.items())[0]}")
    return text_dict


def get_image_index(image_folder: str, show_summary: bool = False) -> dict[str, object]:
    """Load the image files from the folder and return the dictionary of the image content.
    Args:
        image_folder: The folder containing the image files.
        show_summary: Whether to print the summary of the image files.
    Returns:
        The dictionary of the image content. Key: image_id, Value: image_file
    """
    # Load image index from <image_folder>/images.csv, the columns are image_id,height,width,path
    image_folder = os.path.expanduser(image_folder)
    filename_id_mapping = {}
    with open(os.path.join(image_folder, "images.csv"), "r") as f:
        for line in f:
            image_id, _, _, path = line.strip().split(",")
            filename = path.split("/")[-1]
            filename_id_mapping[filename] = image_id
    print(f"Found {len(filename_id_mapping)} image to file mapping in the folder.")
    print(f"Example mapping: {list(filename_id_mapping.items())[0]}")

    image_dict = {}
    files = {}
    for folder, dirs, files in os.walk(image_folder):
        for filename in files:
            if not filename.endswith(".jpg"):
                continue
            image_id = filename_id_mapping.get(filename)
            image_dict[image_id] = os.path.join(folder, filename)

    if show_summary:
        print(f"Found {len(image_dict)} image files in the folder.")
        print(f"Example image file: {list(image_dict.items())[0]}")
    return image_dict


def merge_text_image(text_dict: dict[str, object], image_dict: dict[str, object]) -> dict[str, object]:
    """Merge the text and image dictionaries into one dictionary,
    the key is the image_id, the value is the item_name and the image file name tuple.

    Args:
        text_dict: The dictionary of the text content.
        image_dict: The dictionary of the image content.

    Returns:
        The merged dictionary of the text and image content. Key: image_id, Value: (item_name, image_file)
    """
    merged_dict = {}
    for item_id in text_dict:
        item_name = text_dict[item_id]["item_name"]
        image_ids = text_dict[item_id]["image_ids"]
        for image_id in image_ids:
            if image_id in image_dict:
                image_file = image_dict[image_id]
                merged_dict[image_id] = (item_name, image_file)
    return merged_dict


def load_images_as_batch(image_paths, transform=None):
    """
    Loads images from file paths, applies transformations, and stacks them into a batch tensor.

    Args:
        image_paths (list of str): List of image file paths.
        transform (callable, optional): Transformations to apply to each image.

    Returns:
        torch.Tensor: Batched tensor of images.
    """
    images = []
    for file_path in image_paths:
        try:
            image = Image.open(file_path).convert("RGB")  # Ensure the image is in RGB mode
            if transform:
                image = transform(image)
            images.append(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image at {file_path}: {e}")
    
    # Stack into a single tensor
    batch = torch.stack(images, dim=0)
    return batch
