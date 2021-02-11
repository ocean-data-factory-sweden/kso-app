FROM python:3.7-slim

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

RUN ls -l

EXPOSE 8501

CMD ["streamlit", "run", "app_frontend.py", "--server.port",  "$PORT"]