import os
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL.Image
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Folder of images")
parser.add_argument("--output", type=str, required=True, help="Output folder")
parser.add_argument("--label", type=str, required=False, default="selected_tags.csv", help="By default assumes that 'selected_tags.csv' is on the same directory.")
parser.add_argument("--model", type=str, required=False, default="model.onnx", help="By default assumes that the 'model.onnx' is on the same directory.")
parser.add_argument("--general_score", type=float, required=False, default="0.5", help="Sets the minimum score of 'confidence'. Default '0.5'")
parser.add_argument("--character_score", type=float, required=False, default="0.85", help="Sets the minimum score of 'character confidence'. Default '0.85'")
args = parser.parse_args()

MODEL_FILENAME = args.model
LABEL_FILENAME = args.label
IMAGES_DIRECTORY = args.input
OUTPUT_DIRECTORY = args.output
SCORE_GENERAL_THRESHOLD = args.general_score
SCORE_CHARACTER_THRESHOLD = args.character_score

def load_model(model_filename: str) -> rt.InferenceSession:
    model = rt.InferenceSession(model_filename)
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

    return f"{a}\n"

def main():
    model = load_model(MODEL_FILENAME)
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
    main()