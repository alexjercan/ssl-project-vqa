import os
import wget
import json
import torch
import zipfile
import requests
import wikipedia
import html2text
from torch.utils.data import DataLoader

import pandas as pd

from functools import partial
from datasets import Dataset
from tqdm import tqdm
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)

P = 8
tqdm.pandas()

data_path = "../data"
root_path = os.path.join(data_path, "okvqa")

test_questions_path = os.path.join(root_path, "OpenEnded_mscoco_val2014_questions.json")
test_annotations_path = os.path.join(root_path, "mscoco_val2014_annotations.json")
test_image_path = os.path.join(root_path, "val2014")
test_image_name_prefix = "COCO_val2014_"
test_qa_data_path = os.path.join(root_path, "val2014_qa_data.json")
test_ans_data_path = os.path.join(root_path, "val2014_ans_data.json")

train_questions_path = os.path.join(
    root_path, "OpenEnded_mscoco_train2014_questions.json"
)
train_annotations_path = os.path.join(root_path, "mscoco_train2014_annotations.json")
train_image_path = os.path.join(root_path, "train2014")
train_image_name_prefix = "COCO_train2014_"
train_qa_data_path = os.path.join(root_path, "train2014_qa_data.json")

train_images_url = "http://images.cocodataset.org/zips/train2014.zip"
test_images_url = "http://images.cocodataset.org/zips/val2014.zip"

train_question_url = "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip"
test_questions_url = (
    "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip"
)

train_annotations_url = (
    "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip"
)
test_annotations_url = (
    "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip"
)


urls = [
    train_images_url,
    test_images_url,
    train_question_url,
    test_questions_url,
    train_annotations_url,
    test_annotations_url,
]

zip_paths = list(map(lambda url: os.path.join(data_path, url.split("/")[-1]), urls))


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_zip(url, path, force: bool = False):
    if os.path.exists(path):
        print(f"{path} found. skipping...")
        return

    wget.download(url, out=path, bar=bar_progress)


def download_okvqa(force: bool = False) -> None:
    if os.path.exists(root_path) and not force:
        print("Dataset root dir found. skiping...")
        return

    os.makedirs(root_path, exist_ok=True)

    for url, path in zip(urls, zip_paths):
        download_zip(url, path, force)

    for path in zip_paths:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(root_path)


def read_image(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    return i_image


def get_context_wikipedia(caption):
    try:
        return wikipedia.page(caption).content
    except:
        return ""


def get_caption(
    ic_model,
    ic_feature_extractor,
    ic_tokenizer,
    image_path,
    max_length=16,
    num_beams=4,
):
    image = read_image(image_path)

    pixel_values = ic_feature_extractor(
        images=[image], return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(ic_model.device)

    output_ids = ic_model.generate(
        pixel_values, max_length=max_length, num_beams=num_beams
    )

    preds = ic_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def vqa_to_qa(force: bool = False):
    if os.path.exists(test_qa_data_path) and not force:
        print("QA data found. skipping...")
        return

    with open(test_questions_path, "r") as f:
        test_questions_df = pd.DataFrame(json.load(f)["questions"])

    with open(test_annotations_path, "r") as f:
        test_annotations_df = pd.DataFrame(json.load(f)["annotations"])

    test_df = test_questions_df.merge(test_annotations_df)
    test_df["image_path"] = test_df["image_id"].progress_apply(
        lambda image_id: os.path.join(
            test_image_path, f"{test_image_name_prefix}{image_id:012d}.jpg"
        )
    )

    ic_model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    ic_feature_extractor = ViTFeatureExtractor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    ic_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic_model.to(device)

    test_df["caption"] = test_df["image_path"].progress_apply(
        partial(get_caption, ic_model, ic_feature_extractor, ic_tokenizer)
    )

    test_df["context"] = test_df["caption"].progress_apply(get_context_wikipedia)

    test_df.to_json(test_qa_data_path)


def get_answer(qa_pipeline, row):
    question = row["question"]
    context = row["context"] if row["context"] else row["caption"]

    return qa_pipeline(question, context)


def predict_qa(force: bool = False):
    if os.path.exists(test_ans_data_path) and not force:
        print("Answer data found. skipping...")
        return

    test_qa_df = pd.read_json(test_qa_data_path)

    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2"
    )
    qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=int(torch.cuda.is_available())-1)

    result = pd.DataFrame(test_qa_df[["question", "context", "caption"]].progress_apply(
        partial(get_answer, qa_pipeline), axis=1
    ).tolist())

    test_qa_df["answer"] = result["answer"]

    test_qa_df.to_json(test_ans_data_path)


if __name__ == "__main__":
    download_okvqa()
    vqa_to_qa()
    predict_qa()
