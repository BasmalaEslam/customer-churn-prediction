FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python app.py train
ENV APP_HOST=0.0.0.0
EXPOSE 8101
CMD ["python", "app.py", "serve"]
