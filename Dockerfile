FROM python:3.11.4

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ui.py"]