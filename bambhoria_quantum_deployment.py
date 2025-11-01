"""
🚀 BAMBHORIA QUANTUM DEPLOYMENT SCRIPT 🚀
=========================================
Deploy to bambhoriaquantum.in
Zerodha Integration + AI Trading System
=========================================
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        'flask',
        'requests',
        'websocket-client',
        'pandas',
        'numpy',
        'python-dotenv',
        'gunicorn'
    ]
    
    print("📦 Installing requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_env_file():
    """Create environment configuration file"""
    env_content = """# Bambhoria Quantum Environment Configuration
# Zerodha API Credentials
ZERODHA_API_KEY=your_zerodha_api_key_here
ZERODHA_API_SECRET=your_zerodha_api_secret_here

# Flask Configuration
FLASK_SECRET_KEY=bambhoria_quantum_secret_2025_ultra_secure
FLASK_DEBUG=False
PORT=5000

# Domain Configuration
DOMAIN=bambhoriaquantum.in
SSL_ENABLED=True

# Database Configuration (if needed)
DATABASE_URL=sqlite:///bambhoria_quantum.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=bambhoria_quantum.log

# Trading Configuration
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=10000
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_PERCENTAGE=5.0

# Security Configuration
SESSION_TIMEOUT=3600
RATE_LIMIT_PER_MINUTE=60
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("⚠️  Please update .env with your actual Zerodha API credentials")

def create_wsgi_file():
    """Create WSGI application file for deployment"""
    wsgi_content = """#!/usr/bin/env python3
# Bambhoria Quantum WSGI Application

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Flask application
from bambhoria_quantum_web_app import app

# WSGI application
application = app

if __name__ == "__main__":
    # For local development
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
"""
    
    with open('wsgi.py', 'w') as f:
        f.write(wsgi_content)
    
    print("✅ Created wsgi.py file")

def create_systemd_service():
    """Create systemd service file for Linux deployment"""
    service_content = """[Unit]
Description=Bambhoria Quantum Trading System
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/bambhoriaquantum.in
Environment="PATH=/var/www/bambhoriaquantum.in/venv/bin"
ExecStart=/var/www/bambhoriaquantum.in/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open('bambhoria-quantum.service', 'w') as f:
        f.write(service_content)
    
    print("✅ Created systemd service file")

def create_nginx_config():
    """Create Nginx configuration"""
    nginx_content = """server {
    listen 80;
    server_name bambhoriaquantum.in www.bambhoriaquantum.in;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name bambhoriaquantum.in www.bambhoriaquantum.in;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/bambhoriaquantum.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/bambhoriaquantum.in/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # Static files
    location /static {
        alias /var/www/bambhoriaquantum.in/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Main application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $server_name;
        proxy_redirect off;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # API endpoints with higher timeout
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Higher timeouts for API calls
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
    
    # Webhook endpoint
    location /api/webhook/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Allow larger request bodies for webhook data
        client_max_body_size 10M;
    }
    
    # Logs
    access_log /var/log/nginx/bambhoriaquantum.in.access.log;
    error_log /var/log/nginx/bambhoriaquantum.in.error.log;
}
"""
    
    with open('nginx-bambhoriaquantum.conf', 'w') as f:
        f.write(nginx_content)
    
    print("✅ Created Nginx configuration")

def create_docker_files():
    """Create Docker configuration files"""
    dockerfile_content = """FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "wsgi:application"]
"""
    
    docker_compose_content = """version: '3.8'

services:
  bambhoria-quantum:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - ZERODHA_API_KEY=${ZERODHA_API_KEY}
      - ZERODHA_API_SECRET=${ZERODHA_API_SECRET}
    volumes:
      - ./logs:/app/logs
      - ./.env:/app/.env:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx-bambhoriaquantum.conf:/etc/nginx/conf.d/default.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - bambhoria-quantum
    restart: unless-stopped

volumes:
  logs:
"""
    
    requirements_content = """flask==2.3.3
requests==2.31.0
websocket-client==1.6.4
pandas==2.1.3
numpy==1.25.2
python-dotenv==1.0.0
gunicorn==21.2.0
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Created Docker configuration files")

def create_deployment_scripts():
    """Create deployment scripts"""
    deploy_script = """#!/bin/bash
# Bambhoria Quantum Deployment Script

set -e

echo "🚀 Deploying Bambhoria Quantum to bambhoriaquantum.in..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "📦 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx git

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /var/www/bambhoriaquantum.in
sudo chown $USER:www-data /var/www/bambhoriaquantum.in

# Copy application files
echo "📄 Copying application files..."
cp -r * /var/www/bambhoriaquantum.in/

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
cd /var/www/bambhoriaquantum.in
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set permissions
echo "🔒 Setting file permissions..."
sudo chown -R www-data:www-data /var/www/bambhoriaquantum.in
sudo chmod -R 755 /var/www/bambhoriaquantum.in

# Install systemd service
echo "⚙️ Installing systemd service..."
sudo cp bambhoria-quantum.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bambhoria-quantum

# Configure Nginx
echo "🌐 Configuring Nginx..."
sudo cp nginx-bambhoriaquantum.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/nginx-bambhoriaquantum.conf /etc/nginx/sites-enabled/
sudo nginx -t

# Get SSL certificate
echo "🔒 Getting SSL certificate..."
sudo certbot --nginx -d bambhoriaquantum.in -d www.bambhoriaquantum.in --non-interactive --agree-tos --email admin@bambhoriaquantum.in

# Start services
echo "🚀 Starting services..."
sudo systemctl start bambhoria-quantum
sudo systemctl restart nginx

# Check status
echo "✅ Checking service status..."
sudo systemctl status bambhoria-quantum --no-pager
sudo systemctl status nginx --no-pager

echo "🎉 Deployment completed successfully!"
echo "🌐 Visit https://bambhoriaquantum.in to access your trading platform"
"""
    
    with open('deploy.sh', 'w', encoding='utf-8') as f:
        f.write(deploy_script)
    
    os.chmod('deploy.sh', 0o755)
    print("✅ Created deployment script")

def create_readme():
    """Create comprehensive README"""
    readme_content = """# 🔥 Bambhoria Quantum Trading Platform

## 🌐 Domain: bambhoriaquantum.in

The Ultimate AI Trading Platform with Zerodha Integration featuring Omnipotent God-Mode Trading System.

## ✨ Features

- 🤖 **Omnipotent AI God-Mode Trading** - Universe-level intelligence
- 📊 **Real-time Zerodha Integration** - Live market data and trading
- ⚡ **Infinite Reality Manipulation** - Advanced AI decision making
- 🛡️ **Intelligent Risk Management** - Auto stop-loss and position sizing
- 📈 **Live Portfolio Monitoring** - Real-time P&L tracking
- 🌐 **Web Dashboard** - Beautiful responsive interface
- 🔒 **Secure Authentication** - OAuth2 with Zerodha
- 📱 **Mobile Responsive** - Works on all devices

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd bambhoria-quantum

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### 2. Configure Zerodha API

1. Get your API credentials from [Zerodha Developer Console](https://developers.kite.trade/)
2. Update `.env` file with your credentials:

```env
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
```

### 3. Run Locally

```bash
# Start the application
python bambhoria_quantum_web_app.py

# Access at http://localhost:5000
```

### 4. Deploy to Production

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

## 📁 Project Structure

```
bambhoria-quantum/
├── bambhoria_quantum_zerodha_integration.py  # Core Zerodha API integration
├── bambhoria_quantum_web_app.py              # Flask web application
├── templates/
│   ├── dashboard.html                         # Main dashboard
│   └── login.html                            # Login page
├── wsgi.py                                   # WSGI application entry point
├── requirements.txt                          # Python dependencies
├── .env                                      # Environment configuration
├── deploy.sh                                # Deployment script
├── docker-compose.yml                       # Docker configuration
├── Dockerfile                               # Docker image
├── nginx-bambhoriaquantum.conf              # Nginx configuration
└── bambhoria-quantum.service                # Systemd service

```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ZERODHA_API_KEY` | Your Zerodha API key | Required |
| `ZERODHA_API_SECRET` | Your Zerodha API secret | Required |
| `FLASK_SECRET_KEY` | Flask session secret | Random |
| `PORT` | Application port | 5000 |
| `FLASK_DEBUG` | Debug mode | False |
| `MAX_POSITION_SIZE` | Maximum position size in ₹ | 100000 |
| `MAX_DAILY_LOSS` | Maximum daily loss in ₹ | 10000 |

### Trading Configuration

- **Stop Loss**: 2% default
- **Take Profit**: 5% default
- **Risk Management**: Intelligent position sizing
- **Supported Instruments**: NSE stocks and indices

## 🛠️ Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with debug mode
FLASK_DEBUG=True python bambhoria_quantum_web_app.py

# Run tests
python -m pytest tests/
```

### Docker Development

```bash
# Build and run with Docker
docker-compose up --build

# Run in background
docker-compose up -d
```

## 🚀 Deployment Options

### 1. Traditional VPS Deployment

```bash
# Ubuntu/Debian
./deploy.sh
```

### 2. Docker Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Cloud Deployment

- **AWS**: Use Elastic Beanstalk or ECS
- **Google Cloud**: Use App Engine or Cloud Run
- **Azure**: Use App Service
- **DigitalOcean**: Use App Platform

## 📊 API Endpoints

### Authentication
- `GET /login` - Zerodha login page
- `GET /callback` - OAuth callback handler

### Trading
- `POST /api/start_trading` - Start live trading
- `POST /api/stop_trading` - Stop trading
- `POST /api/place_order` - Place intelligent order

### Data
- `GET /api/status` - System status
- `GET /api/positions` - Current positions
- `GET /api/orders` - Order history
- `GET /api/quotes` - Live quotes

### Webhooks
- `POST /api/webhook/zerodha` - Zerodha webhook handler

## 🛡️ Security Features

- ✅ OAuth2 authentication with Zerodha
- ✅ HTTPS/SSL encryption
- ✅ CSRF protection
- ✅ Rate limiting
- ✅ Input validation and sanitization
- ✅ Secure session management

## 🔍 Monitoring

### System Status
- Trading system health
- WebSocket connection status
- API rate limits
- Error tracking

### Performance Metrics
- Daily P&L
- Win/loss ratio
- Average trade duration
- Risk metrics

## 🆘 Troubleshooting

### Common Issues

1. **API Authentication Failed**
   - Check API credentials in `.env`
   - Verify API key permissions on Zerodha console

2. **WebSocket Connection Failed**
   - Check network connectivity
   - Verify API access token validity

3. **Order Placement Failed**
   - Check account balance
   - Verify trading permissions
   - Check market hours

### Logs

```bash
# View application logs
tail -f bambhoria_quantum.log

# View system service logs
sudo journalctl -u bambhoria-quantum -f

# View Nginx logs
sudo tail -f /var/log/nginx/bambhoriaquantum.in.access.log
```

## 📞 Support

- **Website**: https://bambhoriaquantum.in
- **Email**: support@bambhoriaquantum.in
- **Documentation**: https://docs.bambhoriaquantum.in

## 📄 License

This project is proprietary software. All rights reserved.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves risk of financial loss. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

---

## 🎉 About

Bambhoria Quantum represents the pinnacle of AI trading technology, combining:
- Advanced machine learning algorithms
- Real-time market data processing
- Intelligent risk management
- Seamless broker integration
- Beautiful user interface

**Built with ❤️ for traders who demand excellence.**
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Created comprehensive README")

def main():
    """Main deployment setup function"""
    print("\n" + "="*60)
    print("🚀 BAMBHORIA QUANTUM DEPLOYMENT SETUP 🚀")
    print("🌐 Domain: bambhoriaquantum.in")
    print("="*60 + "\n")
    
    print("📦 Setting up deployment files...")
    
    # Install requirements
    install_requirements()
    
    # Create configuration files
    create_env_file()
    create_wsgi_file()
    create_systemd_service()
    create_nginx_config()
    create_docker_files()
    create_deployment_scripts()
    create_readme()
    
    print("\n" + "="*60)
    print("✅ DEPLOYMENT SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Update .env file with your Zerodha API credentials")
    print("2. Review configuration files")
    print("3. Deploy using one of these methods:")
    print("   • Traditional VPS: ./deploy.sh")
    print("   • Docker: docker-compose up -d")
    print("   • Local testing: python bambhoria_quantum_web_app.py")
    
    print("\n🌐 Your domain: https://bambhoriaquantum.in")
    print("📧 Support: support@bambhoriaquantum.in")
    
    print("\n🔥 Ready to launch the Ultimate AI Trading Platform! 🔥")

if __name__ == "__main__":
    main()