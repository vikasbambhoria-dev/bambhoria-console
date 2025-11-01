#!/bin/bash
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
