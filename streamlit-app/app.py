import streamlit as st

from inference import run, prepare_models
from PIL import Image
from io import BytesIO
from urllib.request import urlopen


@st.cache(allow_output_mutation=True)
def load_model():
    return prepare_models()


def read_image(image_file):
    bytes_data = image_file.getvalue()
    image = Image.open(BytesIO(bytes_data))
    return image


def context_to_html(context, answer):
    start = answer["start"]
    end = answer["end"]

    context_before = context[:start]
    answer_text = context[start:end]
    context_after = context[end:]

    return "<div>" "Context: " + context_before + f"<span style='color:green'>{answer_text}</span>" + context_after + "</div>"


def main():
    ic_feature_extractor, ic_model, ic_tokenizer, qa_model, qa_tokenizer = load_model()

    st.title("Visual Question Answering")

    st.write(
        """
    This is a demo of the Visual Question Answering model.
    It takes an image and a question and returns the answer.
    """
    )

    col1, col2 = st.columns(2)

    image_file = col1.file_uploader("Upload an image", type=["jpg", "png"])
    image_url = col2.text_input("Enter image url:")
    image = read_image(image_file) if image_file else None
    if image is None and image_url:
        with urlopen(image_url) as url:
            image = Image.open(url)

    question = st.text_input("Question")

    if st.button("Answer"):
        caption, answer, context = run(
            ic_feature_extractor,
            ic_model,
            ic_tokenizer,
            qa_model,
            qa_tokenizer,
            image,
            question,
        )

        st.image(image, caption=caption)
        st.write(f"Answer: {answer['answer']} Score: {answer['score']}")
        st.markdown(context_to_html(context, answer), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
