import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList
model_name = "vicuna-7B-1.1-HF"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded")


def predict(query: str, max_tokens: str, top_k: int, temperature: float = 0.8, repetition_penalty: float = 1.1):
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=True,
        top_k=top_k,
        early_stopping=True,
        max_new_tokens=max_tokens,
        temperature=temperature,
        eos_token_id=model.config.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    model.generation_config = generation_config
    inputs = tokenizer(
        query, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        inputs,
        return_dict_in_generate=True,
        output_scores=True)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores,  normalize_logits=True
    )
    input_length = 1 if model.config.is_encoder_decoder else inputs.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    tokens = [tokenizer.decode(v) for v in generated_tokens[0] if v > 2]
    string = tokenizer.decode(generated_tokens[0])
    string = string.replace('</s>', '').replace('<s>', '')
    probs = np.exp(transition_scores.detach().cpu().numpy())[0]

    return tokens, string, probs


if __name__ == "__main__":
    predict("Cual es la capital de Argentina?")
