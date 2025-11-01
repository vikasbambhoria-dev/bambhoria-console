#!/usr/bin/env python3
"""
Simple Dashboard Server for OrderBrain Integration
Provides REST API endpoint for receiving order data
"""

import json
import logging
from aiohttp import web

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("order-dashboard")

async def index(request):
    """Serve the dashboard homepage"""
    return web.FileResponse(path=request.app['static_index'])

async def api_orders(request):
    """Handle incoming order data from OrderBrain"""
    try:
        data = await request.json()
        LOG.info("🧾 Order received: %s", data)
        
        # Add timestamp for tracking
        import time
        data['_received_at'] = int(time.time())
        
        # Store order in memory (for demo purposes)
        request.app['orders'].append(data)
        
        # Keep only last 100 orders
        if len(request.app['orders']) > 100:
            request.app['orders'] = request.app['orders'][-100:]
            
        return web.json_response({"ok": True, "message": "Order processed successfully"})
        
    except Exception as e:
        LOG.error("Error processing order: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=400)

async def api_orders_list(request):
    """Get all stored orders"""
    return web.json_response({
        "orders": request.app['orders'],
        "count": len(request.app['orders'])
    })

def create_app():
    """Initialize the application"""
    app = web.Application()
    app['static_index'] = "static/index.html"
    app['orders'] = []  # In-memory storage for demo
    
    # Routes
    app.router.add_get("/", index)
    app.router.add_post("/api/orders", api_orders)
    app.router.add_get("/api/orders", api_orders_list)
    app.router.add_static("/static/", path="static/", show_index=True)
    
    return app

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001, help="HTTP port")
    args = parser.parse_args()
    
    LOG.info("🚀 Starting OrderBrain Dashboard Server on port %d", args.port)
    app = create_app()
    web.run_app(app, port=args.port)