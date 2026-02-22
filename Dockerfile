FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  openssh-client ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

COPY runpodctl /usr/local/bin/runpodctl
RUN chmod +x /usr/local/bin/runpodctl

RUN npm run build

ENV NODE_ENV=production
CMD ["npm","run","start"]
