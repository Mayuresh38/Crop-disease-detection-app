# Production Deployment & Custom Domain Guide: AgriShield

This guide outlines production deployment methods directly from your GitHub repository, custom domain connection, and automated SSL setup.

---

## 1. Deploy via GitHub to Streamlit Community Cloud (Free & Recommended)

Streamlit Community Cloud connects directly to your GitHub repository and automatically redeploys whenever you push changes to `main`.

### Step-by-Step Instructions:
1. Open your browser and navigate to **[share.streamlit.io](https://share.streamlit.io)**.
2. Click **Continue with GitHub** and authorize with your account (`Mayuresh38`).
3. Click the **"New app"** button.
4. Fill in your repository parameters:
   - **Repository:** `Mayuresh38/Crop-disease-detection-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (subdomain):** e.g., `agrishield-crop-detector.streamlit.app` (or customize as preferred)
5. Click **Deploy!**
   - Streamlit Cloud will read `requirements.txt`, install dependencies, load the serialized model (`models/crop_disease_model.keras`), and launch your live application with free HTTPS.

### Connecting a Custom Domain on Streamlit Cloud:
1. In your deployed app's settings on Streamlit Cloud, go to **Settings > Custom Domain**.
2. Enter your custom domain (e.g., `pathology.yourdomain.com`).
3. In your DNS provider (Cloudflare, GoDaddy, AWS Route 53), add the CNAME record specified by Streamlit pointing to your app's Streamlit domain.

---

## 2. Deploy via GitHub to Render (Web Service)

1. Sign in to **[render.com](https://render.com)** with your GitHub account.
2. Click **New +** > **Web Service**.
3. Select your repository: `Mayuresh38/Crop-disease-detection-app`.
4. Configure service settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Click **Create Web Service**.
6. **Custom Domain:** Go to **Settings > Custom Domains** in Render, add your domain, and configure the CNAME/A record in your DNS host.

---

## 3. Deploy via GitHub to Hugging Face Spaces

1. Create an account at **[huggingface.co](https://huggingface.co)**.
2. Click **New Space**.
3. Select **Space SDK:** `Streamlit`.
4. Connect or link your GitHub repository `Mayuresh38/Crop-disease-detection-app`.
5. Hugging Face builds the container and provides a public endpoint with free GPU/CPU tiers.

---

## 4. Self-Hosted VPS / Cloud Server with Custom Domain & Nginx

To host on your own Linux server (Ubuntu/Debian VPS, AWS EC2, DigitalOcean Droplet):

### Step 1: Clone Repository
```bash
git clone https://github.com/Mayuresh38/Crop-disease-detection-app.git
cd Crop-disease-detection-app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Systemd Background Service
Create `/etc/systemd/system/agrishield.service`:
```ini
[Unit]
Description=AgriShield Crop Disease Decision Engine
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/Crop-disease-detection-app
ExecStart=/var/www/Crop-disease-detection-app/venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
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

### Step 3: Nginx Reverse Proxy & Custom Domain Configuration
Install Nginx and Certbot:
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

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/agrishield /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 4: Automated SSL (HTTPS) with Let's Encrypt
```bash
sudo certbot --nginx -d pathology.yourdomain.com
```

---

## 5. Docker Container Deployment

Build and run the containerized image:
```bash
docker build -t agrishield-detector:latest .
docker run -d --name agrishield -p 8501:8501 --restart unless-stopped agrishield-detector:latest
```
