FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# EXTRAS="[dev]" ставит pytest и ruff (используется сервисом test)
ARG EXTRAS=""

# Установка editable: пути проекта (PROJECT_ROOT) вычисляются от расположения исходников
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e ".${EXTRAS}"

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p data logs reports \
    && chown -R app:app /app
USER app

EXPOSE 8501
CMD ["streamlit", "run", "src/experimentlab/app/dashboard.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
