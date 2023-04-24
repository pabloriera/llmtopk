FROM python:3

RUN pip install fastapi uvicorn gradio llama-cpp-python matplotlib numpy 

WORKDIR /model

COPY . .

RUN python init.py

CMD uvicorn app:app --host 0.0.0.0 --port 9999 --reload --reload-dir .