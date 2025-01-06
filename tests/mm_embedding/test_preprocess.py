"""This file contains the test code for preprocess."""
from ml_prototype.mm_embedding.preprocess import get_text_index, get_image_index, merge_text_image


if __name__ == "__main__":
    text_folder = "~/Downloads/multimodal/abo-listings"
    text_dict = get_text_index(text_folder, True)

    image_folder = "~/Downloads/multimodal/images"
    image_dict = get_image_index(image_folder, True)

    merged_dict = merge_text_image(text_dict, image_dict)
    print(f"Merged {len(merged_dict)} items with text and image information.")
    print("Example item:")
    # Peep into the merged dictionary