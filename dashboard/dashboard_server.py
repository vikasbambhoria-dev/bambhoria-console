
#!/usr/bin/env python3
import asyncio, argparse, os, json, logging, time
from aiohttp import web, ClientSession, WSCloseCode
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("godeye-dashboard")
SRC_WS_DEFAULT = os.environ.get("GODEYE_DATA_WS", "ws://127.0.0.1:8765/ws")
async def index(request):
    return web.FileResponse(path=request.app['static_index'])

async def api_orders(request):
    """Handle incoming order data from OrderBrain"""
    try:
        data = await request.json()
        LOG.info("🧾 Order received: %s", data)
        
        # Add timestamp for tracking
        data['_received_at'] = int(time.time())
        
        # Broadcast order to all connected WebSocket clients
        payload = json.dumps({
            'type': 'order_update',
            'data': data
        })
        
        dead = []
        for client in list(request.app['clients']):
            try:
                await client.send_str(payload)
            except Exception:
                dead.append(client)
        
        # Clean up dead connections
        for d in dead:
            request.app['clients'].discard(d)
            
        return web.json_response({"ok": True, "message": "Order processed"})
        
    except Exception as e:
        LOG.error("Error processing order: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=400)
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['clients'].add(ws)
    LOG.info("Browser connected, total clients=%d", len(request.app['clients']))
    try:
        async for msg in ws:
            pass
    finally:
        request.app['clients'].discard(ws)
        LOG.info("Browser disconnected, total clients=%d", len(request.app['clients']))
    return ws
async def forwarder(app, src_ws):
    LOG.info("Starting forwarder to %s", src_ws)
    session = ClientSession()
    try:
        async with session.ws_connect(src_ws) as ws_src:
            LOG.info("Connected to source WS %s", src_ws)
            async for msg in ws_src:
                if msg.type == 1:
                    payload = msg.data
                    # enrich with server timestamp
                    try:
                        j = json.loads(payload)
                        j['_forwarded_at'] = int(time.time())
                        payload = json.dumps(j)
                    except Exception:
                        pass
                    dead = []
                    for c in list(app['clients']):
                        try:
                            await c.send_str(payload)
                        except Exception:
                            dead.append(c)
                    for d in dead:
                        app['clients'].discard(d)
                else:
                    LOG.debug("Non-text msg from source: %s", msg.type)
    except Exception as e:
        LOG.exception("Forwarder exception: %s", e)
    finally:
        await session.close()
        LOG.info("Forwarder ended, will retry in 3s")
        await asyncio.sleep(3)
        if not app.get('stopping', False):
            app['runner_task'] = app.loop.create_task(forwarder(app, src_ws))
async def on_startup(app):
    app['clients'] = set()
    src_ws = app['src_ws']
    app['runner_task'] = app.loop.create_task(forwarder(app, src_ws))
async def on_cleanup(app):
    app['stopping'] = True
    t = app.get('runner_task')
    if t:
        t.cancel()
    for c in list(app.get('clients', [])):
        try:
            await c.close(code=WSCloseCode.GOING_AWAY)
        except:
            pass
def make_app(static_index, src_ws):
    app = web.Application()
    app['static_index'] = static_index
    app['src_ws'] = src_ws
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/orders", api_orders)
    app.router.add_static("/static/", path=static_index.replace("index.html",""), show_index=True)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--http-port", type=int, default=5000)
    p.add_argument("--src-ws", type=str, default=None)
    args = p.parse_args()
    src_ws = args.src_ws or os.environ.get("GODEYE_DATA_WS", SRC_WS_DEFAULT)
    app = make_app(static_index="static/index.html", src_ws=src_ws)
    web.run_app(app, port=args.http_port)
