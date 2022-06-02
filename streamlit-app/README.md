# Visual Question Answering Container

Docker image using streamlit to test the VQA pipeline.

![thumbnail](./resources/qa_example.png)

## Quickstart

### Docker Compose

```console
docker-compose up
```

### Docker

```console
docker build . --tag visual-question-answering
docker run --it -p 8051:8051 visual-question-answering:latest
```

### Streamlit

```console
pip install -r requirements.txt
streamlit run app.py
```

This will download all the dependencies from `requirements.txt` and the models
from huggingface. Then it will run the Streamlit application on the address:
`localhost:8051`.
