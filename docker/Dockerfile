FROM node:22 AS ts-builder
WORKDIR /build
COPY package*.json tsconfig.json ./
RUN npm ci
COPY lmlab/src ./lmlab/src
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .
COPY lmlab ./lmlab

COPY --from=ts-builder /build/lmlab/dist ./lmlab/dist
EXPOSE 8484

CMD ["uvicorn", "lmlab.app:app", "--host", "0.0.0.0", "--port", "8484"]
