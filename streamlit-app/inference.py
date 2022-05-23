import torch
import random
import requests
import html2text
import wikipedia

from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)
from nltk import pos_tag, word_tokenize

from download_models import download_models


def prepare_models():
    (
        ic_feature_extractor,
        ic_model,
        ic_tokenizer,
        qa_model,
        qa_tokenizer,
    ) = download_models()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qa_model.to(DEVICE)
    ic_model.to(DEVICE)

    return ic_feature_extractor, ic_model, ic_tokenizer, qa_model, qa_tokenizer


def get_wikipedia_page(string):
    try:
        p = wikipedia.summary(string, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        s = e.options[0]
        p = wikipedia.summary(s, auto_suggest=False)

    return p


def get_context_wikipedia(caption):
    words = word_tokenize(caption)
    tags = pos_tag(words)

    tags = [w for (w, t) in tags if t[0] == "N"]

    options = list(set([o for tag in tags for o in wikipedia.search(tag, results=1)]))
    pages = [caption] + [get_wikipedia_page(option) for option in options]

    return "\n".join(pages)


def get_caption(
    ic_model,
    ic_feature_extractor,
    ic_tokenizer,
    image,
    max_length=16,
    num_beams=4,
):
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


def get_answer(qa_model, qa_tokenizer, question, context):
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
    result = qa_pipeline(question, context)

    return result


def run(
    ic_feature_extractor,
    ic_model,
    ic_tokenizer,
    qa_model,
    qa_tokenizer,
    image,
    question,
    max_length=16,
    num_beams=4,
):
    caption = get_caption(
        ic_model,
        ic_feature_extractor,
        ic_tokenizer,
        image,
        max_length=max_length,
        num_beams=num_beams,
    )
    context = get_context_wikipedia(caption)
    answer = get_answer(qa_model, qa_tokenizer, question, context)

    return caption, answer, context
