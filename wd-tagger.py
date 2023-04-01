import os
import re
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL.Image
import cv2
import argparse
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./Test", help="Folder of images. By default will be using the Test folder")
parser.add_argument("--output", type=str, help="Output folder. By default same as the input")
parser.add_argument("--label", type=str, default="./Models/selected_tags.csv", help="By default assumes that 'selected_tags.csv' is on the same directory.")
parser.add_argument("--model", type=str, default="Swinv2", help="By default assumes that the 'model.onnx' is on the same directory. Options: 'convnextv2', 'convnext_tagger_v2', 'swinv2', 'ViTv2'")
parser.add_argument("--general_score", type=float, default="0.5", help="Sets the minimum score of 'confidence'. Default '0.5'")
parser.add_argument("--character_score", type=float, default="0.85", help="Sets the minimum score of 'character confidence'. Default '0.85'")
parser.add_argument("--gpu", action='store_true', help="Use GPU for prediction if available. Faster and useful for large datasets")
parser.add_argument('--add_keyword', type=str, default='', help='keyword to add to output')
args = parser.parse_args()

MODEL_NAME = args.model
LABEL_FILENAME = args.label
IMAGES_DIRECTORY = args.input
OUTPUT_DIRECTORY = args.output
SCORE_GENERAL_THRESHOLD = args.general_score
SCORE_CHARACTER_THRESHOLD = args.character_score
USE_GPU = args.gpu

if args.output is None:
    OUTPUT_DIRECTORY = IMAGES_DIRECTORY
else:
    OUTPUT_DIRECTORY = args.output

MODEL_FILENAME = f"./Models/{MODEL_NAME}.onnx"

def download_file(url: str, filename: str):
    urllib.request.urlretrieve(url, filename)

def download_files():
    model_urls = {
        "convnextv2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx",
        "convnext_tagger_v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx",
        "swinv2": "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/model.onnx",
        "ViTv2": "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2/resolve/main/model.onnx",
    }
    LABEL_URL = "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/resolve/main/selected_tags.csv"
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"Downloading {MODEL_FILENAME}...")
        download_file(model_urls[MODEL_NAME], MODEL_FILENAME)
    
    if not os.path.exists(LABEL_FILENAME):
        print(f"Downloading {LABEL_FILENAME}...")
        download_file(LABEL_URL, LABEL_FILENAME)

def load_model(model_filename: str, use_gpu: bool) -> rt.InferenceSession:
    sess_options = rt.SessionOptions()
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    
    if use_gpu and 'CUDAExecutionProvider' in rt.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    model = rt.InferenceSession(model_filename, sess_options, providers=providers)
    return model

def load_labels() -> list[str]:
    df = pd.read_csv(LABEL_FILENAME)
    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def predict(
    model,
    image: PIL.Image.Image,
    general_threshold: float,
    character_threshold: float,
    tag_names: list[str],
    rating_indexes: list[np.int64],
    general_indexes: list[np.int64],
    character_indexes: list[np.int64],
):
    _, height, width, _ = model.get_inputs()[0].shape
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)
    image = image[:, :, ::-1]
    image_size = (height, height)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    a = (
        ", ".join(list(b.keys()))
        .replace("_", " ")
        .replace("(", "\(")
        .replace(")", "\)")
    )
    c = ", ".join(list(b.keys()))
    character_tags = ', '.join(list(character_res.keys())).replace('_', ' ')
    character_tags = re.sub(r'\([^)]*\)', '', character_tags)
    result = f"{character_tags}, {a}"
    result = re.sub(r'\s*,\s*', ', ', result)
    
    if args.add_keyword:
        return f"{args.add_keyword}, {a}"
    else:
        return result

def main():
    model = load_model(MODEL_FILENAME, USE_GPU)
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels()

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    for filename in os.listdir(IMAGES_DIRECTORY):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue

        print(f"Processing {filename}...")

        image_path = os.path.join(IMAGES_DIRECTORY, filename)
        image = PIL.Image.open(image_path)

        result = predict(
            model,
            image,
            SCORE_GENERAL_THRESHOLD,
            SCORE_CHARACTER_THRESHOLD,
            tag_names,
            rating_indexes,
            general_indexes,
            character_indexes,
        )

        output_path = os.path.join(OUTPUT_DIRECTORY, os.path.splitext(filename)[0] + ".txt")

        with open(output_path, "w") as f:
            f.write(str(result))

if __name__ == "__main__":
    download_files()
    main()