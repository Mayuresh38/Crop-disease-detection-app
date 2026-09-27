# Production Deployment & Custom Domain Guide: AgriShield

This guide outlines production deployment, reverse proxy setup, SSL certificate generation, and custom domain connection procedures for AgriShield.

---

## 1. Custom Domain Connection & DNS Configuration

To bind a custom domain (e.g., `pathology.yourdomain.com` or `yourdomain.com`) to your server instance:

### Step 1: DNS Records Setup
In your DNS provider (Cloudflare, AWS Route 53, Namecheap, GoDaddy, etc.), configure:

| Record Type | Host / Name | Target / Value | TTL |
| :--- | :--- | :--- | :--- |
| **A Record** | `@` (or subdomain `pathology`) | `YOUR_SERVER_PUBLIC_IPV4` | Automatic / 300s |
| **CNAME** (Optional) | `www` | `pathology.yourdomain.com` | Automatic / 300s |

Verify DNS propagation:
```bash
nslookup pathology.yourdomain.com
```

### Step 2: Nginx Reverse Proxy Configuration with Custom Domain
Install and configure Nginx on your Linux server:
```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/agrishield`:
```nginx
server {
    server_name pathology.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Streamlit real-time events
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable the configuration:
```bash
sudo ln -s /etc/nginx/sites-available/agrishield /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 3: Automated SSL Certificate (Let's Encrypt)
Run Certbot to secure your custom domain with HTTPS:
```bash
sudo certbot --nginx -d pathology.yourdomain.com
```
Certbot automatically installs certificates and sets up automatic renewal.

---

## 2. Docker Container Deployment

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Build & Run
```bash
docker build -t agrishield-detector:latest .
docker run -d --name agrishield -p 8501:8501 --restart unless-stopped agrishield-detector:latest
```

---

## 3. Systemd Service Deployment (Bare Metal / VPS)

Create `/etc/systemd/system/agrishield.service`:
```ini
[Unit]
Description=AgriShield Crop Disease Decision Engine
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/crop_disease_detector
ExecStart=/usr/bin/python3 -m streamlit run app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agrishield
sudo systemctl start agrishield
```
