FROM nvidia/cuda:11.7.1-base-ubuntu20.04
ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt install -y python3.9 python3-pip git git-lfs -y


RUN pip install  torch bitsandbytes fastapi uvicorn gradio llama-cpp-python matplotlib numpy transformers datasets peft evaluate rouge_score tokenizers sentencepiece protobuf

RUN pip install --upgrade protobuf==3.20.0

# RUN https://gist.githubusercontent.com/TimDettmers/1f5188c6ee6ed69d211b7fe4e381e713/raw/4d17c3d09ccdb57e9ab7eca0171f2ace6e4d2858/check_bnb_install.py && python check_bnb_install.py


RUN git lfs install

WORKDIR /model

# RUN git clone https://huggingface.co/TheBloke/vicuna-7B-1.1-HF

COPY *.py .

# RUN python init.py

CMD uvicorn app:app --host 0.0.0.0 --port 9999 --reload --reload-dir .