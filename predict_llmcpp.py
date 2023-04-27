import time
from llama_cpp import Llama, llama_cpp
import numpy as np
model_path = "ggml-model-q4_0.bin"
llm = Llama(model_path=model_path, logits_all=True, n_batch=1, n_threads=4)
nvocab = llama_cpp.llama_n_vocab(llm.ctx)


def predict_old(query: str, max_tokens: int = 1, top_k: int = 10):
    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()

    llm.reset()
    tokens = llm.tokenize(query.encode())
    ntokens = 1

    rs = []
    for i in range(max_tokens):
        llm.eval(tokens)
        x = llama_cpp.llama_get_logits(llm.ctx)
        X = np.ctypeslib.as_array(x, shape=(ntokens, nvocab,)).copy()
        topktokens = np.argsort(X[0, :])[-top_k:][::-1]
        tokens.append(topktokens[0])
        topklogits = X[0, topktokens]
        strings = [llm.detokenize([t]).decode() for t in topktokens]
        r = [{"string": s, "prob": p} for s, p in zip(strings, topklogits)]
        rs.append(r)
        print(r)

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print("\n\n========\n")

    return rs


def predict(query: str, max_tokens: str, top_k: int):
    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()
    for k in range(1, top_k+1):
        llm.reset()
        tokens = llm.tokenize(query.encode())
        ntokens = 1
        llm.eval(tokens)
        x = llama_cpp.llama_get_logits(llm.ctx)
        X = np.ctypeslib.as_array(x, shape=(ntokens, nvocab,)).copy()
        token0 = np.argsort(X[0, :])[-k]
        tokens.append(token0)
        completion_tokens = []
        for t in llm.generate(tokens, top_k=1, top_p=0.95, temp=0, repeat_penalty=1.1):
            completion_tokens.append(t)
            if len(completion_tokens) >= max_tokens:
                break
        yield llm.detokenize([token0]+completion_tokens).decode()

    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print("\n\n========\n")


if __name__ == "__main__":
    predict("Cual es la capital de Argentina?")
