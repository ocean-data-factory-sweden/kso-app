# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.05-py3
MAINTAINER Jannes Germishuys jurie.germishuys@combine.se

# Install linux packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&\
    apt-get -y upgrade &&\
    apt-get -y install zip htop screen libgl1-mesa-glx build-essential cmake unzip pkg-config &&\
    apt-get -y install libjpeg-dev libpng-dev libtiff-dev &&\
    apt-get -y install ffmpeg &&\
    apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev &&\
    apt-get -y install libxvidcore-dev libx264-dev &&\
    apt-get -y install libgtk-3-dev &&\
    apt-get -y install libcanberra-gtk* &&\
    apt-get -y install libatlas-base-dev gfortran &&\
    apt-get -y install python3-dev

# Install python dependencies
COPY requirements.txt .
#RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache -r requirements.txt coremltools onnx gsutil notebook paramiko
RUN pip install --no-cache  -U torch torchvision numpy
# RUN pip install --no-cache torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
ADD https://api.github.com/repos/ocean-data-factory-sweden/koster_yolov4/git/refs/heads/master version.json
RUN git clone --recurse-submodules -b master https://github.com/ocean-data-factory-sweden/koster_yolov4.git
WORKDIR /usr/src/app/koster_yolov4
COPY fast_yolov4.py .
COPY requirements.txt .

RUN ls -l

# Copy contents
COPY . /usr/src/app

# Set new environment reference
ENV PYTHONPATH "/usr/local/lib/python3.6/site-packages/cv2/python-3.6/:${PYTHONPATH}"
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/koster_yolov4/src"
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/koster_yolov4/"
ENV HOME=/usr/src/app
ENV WANDB_DIR=/data/wandb
ENV WANDB_CACHE_DIR=/data/wandb_cache

# Confirm path
RUN echo $PYTHONPATH

# Install dependencies (pip or conda)
RUN pip install -U -r requirements.txt
RUN pip install --upgrade gsutil
RUN pip install --upgrade protobuf
WORKDIR /usr/src/app/koster_yolov4
COPY kso_utils .

# Run FASTAPI service
EXPOSE 5000
CMD ["uvicorn", "fast_yolov4:app", "--port", "5000", "--host", "0.0.0.0", "--workers", "1"]


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Clean up
# docker system prune -a --volumes
