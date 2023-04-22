import typing

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import matplotlib
import matplotlib.cm as cm
import numpy as np

from predict import predict

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


# --- gradio demo ---

def generate_html_color_map(tokens, probs):
    cmap = cm.get_cmap('winter')
    # Normalize the probabilities to the [0, 1] range
    norm_probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))

    # div flow

    html = ''
    for token, prob in zip(tokens, norm_probs):
        # Get the color corresponding to the probability from the colormap
        color = matplotlib.colors.to_hex(cmap(prob))
        # Add the token with the color as a list item
        html += f'<p style="color:{color}">{token}</p>\n'
    return html


def gradio_predict(question: str, k: int = 1):
    results = predict(question, int(k))
    html = "<div style='display: flex; flex-direction: row;'>"
    for emission in results:
        strings = [t['string'] for t in emission]
        probs = [t['prob'] for t in emission]
        html += "<div style='margin: 10px;'>"
        html += generate_html_color_map(strings, probs)
        html += "</div>"
    html += "</div>"
    return html


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Textbox(
        label="Continua frase", placeholder="Cual es la capital de Argentina?"
    ),
        gr.Number(label="Max Tokens", value=1)
    ],
    outputs=["html"],
    allow_flagging="never",
)

app = gr.mount_gradio_app(app, demo, path="/")
