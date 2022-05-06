FROM python:3.8

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN python download_models.py

CMD ["streamlit", "run", "app.py"]

