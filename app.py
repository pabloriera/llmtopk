from predict import predict
import typing

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import matplotlib
import matplotlib.cm as cm
import numpy as np


def generate_html_color_map(string, tokens, probs):
    cmap = cm.get_cmap('winter')
    html = ""
    for token, prob in zip(tokens, probs):
        color = matplotlib.colors.to_hex(cmap(prob))
        loc = string.find(token)
        if loc + len(token) < len(string) and string[loc + len(token)] == " ":
            lt = 1
        else:
            lt = 0
        # print(string[loc + len(token)])
        p, s, string = [string[:loc], string[loc:loc +
                                             len(token)+lt], string[loc + len(token)+lt:]]
        html += f'<span style="color:{color}">{s}</span>'
    return html


app = FastAPI()

# --- fastapi /predict route ---


class Request(BaseModel):
    question: str


class Result(BaseModel):
    string: str
    prob: float


class Emissions(BaseModel):
    emissions: typing.List[Result]


class Response(BaseModel):
    results: typing.List[Emissions]


@app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    results = predict(request.question)
    return Response(
        results=[
            Emissions(emissions=[
                Result(string=r["string"], prob=r["prob"])
                for r in emission
            ])
            for emission in results
        ]
    )


html_pre = ""


def gradio_predict(question: str,
                   max_tokens: int = 20,
                   k: int = 1):
    global html_pre
    tokens, string, probs = predict(question,
                                    int(max_tokens),
                                    int(k))
    print(question, string, probs)
    html = "<div>"
    html += "<p style='margin: 10px; font-size: 25px;'>"
    html += question+" "
    html += generate_html_color_map(string, tokens, probs)
    html += "</p>"
    html_out = html + "</div>" + html_pre
    html_pre = html_out

    return html_out


demo = gr.Interface(
    css=".gradio-container {background-color: rgb(170, 197, 255);}",
    fn=gradio_predict,
    inputs=[gr.Textbox(
            label="Ingresar texto para continuar", value="¿Cuál es la capital de Argentina?"
            ),
            gr.Number(label="Max Tokens", value=20),
            gr.Number(label="Top k", value=1)
            ],
    outputs=gr.HTML(),
    allow_flagging="never",
).queue()

app = gr.mount_gradio_app(app, demo, path="/",)
