FROM svizor/zoomcamp-model:mlops-3.10.0-slim

# do stuff here
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "starter.py", "./"]

RUN pip install pipenv

RUN pipenv install --system

CMD ["python", "starter.py", "2022", "04"]