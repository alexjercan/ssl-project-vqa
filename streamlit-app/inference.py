import requests
import html2text

from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline,
)

from googlesearch import search


def prepare_models():
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2"
    )
    qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    ic_model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    ic_feature_extractor = ViTFeatureExtractor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    ic_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    return ic_feature_extractor, ic_model, ic_tokenizer, qa_model, qa_tokenizer


def get_context_google(caption, num_results=5):
    links = list(search(caption, num_results=num_results))

    html_conv = html2text.HTML2Text()
    html_conv.ignore_links = True
    html_conv.escape_all = True

    text = []
    for link in links:
        req = requests.get(link)
        text.append(html_conv.handle(req.text))

    return " ".join(text)


def ic_predict_step(
    ic_feature_extractor, ic_model, ic_tokenizer, image, max_length=16, num_beams=4
):
    pixel_values = ic_feature_extractor(
        images=[image], return_tensors="pt"
    ).pixel_values
    # pixel_values = pixel_values.to(device)

    output_ids = ic_model.generate(
        pixel_values, max_length=max_length, num_beams=num_beams
    )

    preds = ic_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]


def qa_predict_step(qa_model, qa_tokenizer, question, context, max_length=16):
    nlp = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
    answer = nlp(question, context)["answer"]

    return answer


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
    caption = ic_predict_step(
        ic_feature_extractor,
        ic_model,
        ic_tokenizer,
        image,
        max_length=max_length,
        num_beams=num_beams,
    )
    context = get_context_google(caption)
    answer = qa_predict_step(
        qa_model, qa_tokenizer, question, context, max_length=max_length
    )

    return caption, answer
