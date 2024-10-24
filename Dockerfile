FROM python:3.11.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]