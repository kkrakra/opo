FROM cr.msk.sbercloud.ru/aicloud-base-images/horovod-cuda10.1-tf2.2.0

USER root
ARG DEBIAN_FRONTEND=noninteractive

USER user

RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace