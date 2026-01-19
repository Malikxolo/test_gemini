import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_test")

async def test_connection():
    uri = "wss://voice-backend-m6w6.onrender.com/ws"
    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connection successful!")
            response = await websocket.recv()
            logger.info(f"Received: {response}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
