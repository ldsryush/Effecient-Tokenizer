# Publishing to Docker Hub

Run these commands once to make the image publicly available so anyone can
start the proxy with a single `docker run` command.

---

## Step 1 — Create a Docker Hub account

Go to https://hub.docker.com and create a free account.
Your username will appear in the image name: `yourusername/efficient-tokenizer`.

---

## Step 2 — Log in from your terminal

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

---

## Step 3 — Build the image

Run this from the repo root (where the `Dockerfile` lives):

```bash
docker build -t ldsryush/efficient-tokenizer:latest .
docker build -t ldsryush/efficient-tokenizer:2.0.0 .
```

> Tag with both `latest` and the version number so users can pin to a specific version.

---

## Step 4 — Push to Docker Hub

```bash
docker push ldsryush/efficient-tokenizer:latest
docker push ldsryush/efficient-tokenizer:2.0.0
```

The push takes 1–3 minutes. After it completes, the image is live at:
https://hub.docker.com/r/ldsryush/efficient-tokenizer

---

## Step 5 — Verify it works

Test the published image from scratch (simulates what the trial user experiences):

```bash
docker run -d \
  --name et-test \
  -p 8001:8000 \
  -e DISPATCH_DRY_RUN=true \
  ldsryush/efficient-tokenizer:latest

curl http://localhost:8001/health
# Expected: {"status":"ok","store":true,"version":"2.0.0"}

docker stop et-test && docker rm et-test
```

---

## What to send the trial company

Once the image is pushed, send them `TRIAL.md` (already in this repo).
The only command they need to run is:

```bash
docker run -d \
  --name efficient-tokenizer \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-their-key-here" \
  ldsryush/efficient-tokenizer:latest
```

Then they open http://localhost:8000/dashboard in a browser.

---

## Updating the image after code changes

Every time you push new code, rebuild and re-push:

```bash
docker build -t ldsryush/efficient-tokenizer:latest .
docker push ldsryush/efficient-tokenizer:latest
```

Trial users get the update by running:
```bash
docker pull ldsryush/efficient-tokenizer:latest
docker stop efficient-tokenizer && docker rm efficient-tokenizer
docker run -d --name efficient-tokenizer -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." ldsryush/efficient-tokenizer:latest
```

---

## Optional: Add a Docker Hub README

Docker Hub shows a README on the image page. Copy the content of `TRIAL.md`
into the "Overview" field at:
https://hub.docker.com/r/ldsryush/efficient-tokenizer

This means anyone who finds the image on Docker Hub gets the full setup guide.
