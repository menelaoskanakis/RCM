FROM nvcr.io/nvidia/pytorch:19.12-py3

RUN pip install --upgrade pip
RUN pip install \
    Pillow==6.2.1 \
    scipy==1.2.0
