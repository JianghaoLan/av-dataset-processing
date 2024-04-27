FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# COPY requirements.txt /temp/
# WORKDIR /temp

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update -y && apt-get install -y ffmpeg

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
&& pip install torch torchvision numpy scipy scikit-image tqdm matplotlib opencv-python-headless yt-dlp

WORKDIR /app

CMD ["bash"]
