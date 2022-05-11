from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

QA_MODEL_NAME = "deepset/roberta-base-squad2"
IC_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


def download_models():
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL_NAME,
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)

    ic_model = VisionEncoderDecoderModel.from_pretrained(
        IC_MODEL_NAME,
    )
    ic_feature_extractor = ViTFeatureExtractor.from_pretrained(
        IC_MODEL_NAME,
    )
    ic_tokenizer = AutoTokenizer.from_pretrained(IC_MODEL_NAME)

    return ic_feature_extractor, ic_model, ic_tokenizer, qa_model, qa_tokenizer


if __name__ == "__main__":
    download_models()
