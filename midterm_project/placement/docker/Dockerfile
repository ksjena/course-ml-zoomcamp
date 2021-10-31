FROM python:3.8.12-slim
WORKDIR /app
RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --system
COPY predict.py model.bin ./
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]