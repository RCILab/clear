FROM python:3.11-slim

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "unittest", "tests.test_core", "tests.test_smg", "tests.test_timing_v2", "-v"]
