import asyncio
from voice.text_to_speech import speak


async def async_speak(text):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, speak, text)


async def handle_actions(action_queue):
    loop = asyncio.get_event_loop()

    while True:
        action = await loop.run_in_executor(None, action_queue.get)

        if action["type"] == "speak":
            await async_speak(action["text"])

        elif action["type"] == "move":
            print("Moving:", action)

        await asyncio.sleep(0)