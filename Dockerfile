FROM python:3.7-slim

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

RUN ls -l

EXPOSE 8501

CMD ["sh", "-c", "streamlit", "run", "app_frontend.py"]