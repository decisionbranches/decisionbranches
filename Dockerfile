FROM python:3.8
WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

CMD [ "python", "./examples/pipeline.py" ]
