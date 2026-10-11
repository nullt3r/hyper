FROM python:3.12-slim-bookworm
# Google Chrome driven over CDP by SeleniumBase (UC / CDP mode), headful in Xvfb: passes Cloudflare / Turnstile checks
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget ca-certificates xvfb xauth scrot tk python3-tk fonts-liberation fonts-dejavu-core fonts-noto-color-emoji \
    && wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb && rm -f /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir seleniumbase pyautogui httpx trafilatura "mcp[cli]>=1.10,<2" && sbase get uc_driver
COPY server.py /app/server.py
WORKDIR /app
CMD ["python", "/app/server.py"]
