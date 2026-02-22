FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  openssh-client ca-certificates wget \
  && rm -rf /var/lib/apt/lists/*

# Install runpodctl (linux-amd64)
RUN wget --quiet https://github.com/Run-Pod/runpodctl/releases/download/v1.14.3/runpodctl-linux-amd64 \
      -O /usr/bin/runpodctl \
  && chmod +x /usr/bin/runpodctl

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

RUN npm run build

ENV NODE_ENV=production
CMD ["npm","run","start"]
