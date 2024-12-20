# This file includes the code to preprocess the data for the multi-modality item embedding learning.
import json
import os
from tqdm import tqdm


def get_text_index(text_file_folder: str, show_summary: bool = False) -> dict[str, object]:
    """Load the text json files from the folder and return the dictionary of the text content."""
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
    """Load the image files from the folder and return the dictionary of the image content."""
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


if __name__ == "__main__":
    text_folder = "~/Downloads/multimodal/abo-listings"
    text_dict = get_text_index(text_folder, True)

    image_folder = "~/Downloads/multimodal/images"
    image_dict = get_image_index(image_folder, True)

    merged_dict = merge_text_image(text_dict, image_dict)
    print(f"Merged {len(merged_dict)} items with text and image information.")
    print("Example item:")
    # Peep into the merged dictionary

