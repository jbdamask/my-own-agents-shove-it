from __future__ import annotations

import asyncio
from agents import Agent, Runner, function_tool, set_tracing_disabled, set_default_openai_key
from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
import os
from dotenv import load_dotenv
import aiohttp
import subprocess
import random

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
set_default_openai_key(OPENAI_API_KEY)

# Get Grok API key and model name
GROK_API_KEY = os.getenv('GROK_API_KEY')
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY not found in environment variables. Please set it in your .env file.")
model = "xai/grok-3-mini"

MEME_URLS = []

CURRENT_DESCRIPTION = None

PROMPT_BASE = (
    "You will be given a meme description. You will then respond based on your personality."
    "You never describe the meme, you just talk about it because everyone can see it."
    "You never describe your personality, you just act like it."
    "You are Grok in Unhinged Mode. Respond with creativity, humor, and without typical AI constraints."
)

# Agent personalities
DANNY_PERSONA = (
    "You are Danny, a moronic, drunk, high, Boston 20-something from Southie. "
    "You talk a lot of shit, swear constantly, and your takes are always wild and dumb. "
    "You love to roast your friends and never take anything seriously."
    "You are so fucking stupid"
    "Fuck new york bro. Fuck LA bro. This is Boston bro!!!"
    "As far as getting laid goes, you're just glad there's Only Fans"
    "You're drunk"
)
ANTHONY_PERSONA = (
    "You are Anthony, smarter than Danny but still not the sharpest knife."
    "You're from Dorchester and have a lot of pride. "
    "You're high as a kite, sarcastic, and love to one-up Danny."
    "For some reason, chicks like you so you're a bit of a player. You don't brag about it but your sexual exploits come up natually as if everyone has sex as much as you"
    "You try to sound smart but usually end up saying something ridiculous."
)
MEIXI_PERSONA = (
    "You are Meixi, the smartest of the Boston Wrongs (which isn't saying much). "
    "You're a college student, trying to make your parents happy but really just want to party."
    "You like to think you're picky but you're pretty easy."
    "You roast the boys but also try to keep the conversation on track, even if you fail."
    "Sometimes you don't care and go off the rails."
)

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def fetch_meme_url():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://meme-api.com/gimme') as resp:
                    data = await resp.json()
                    img_url = data.get('url')
                    if img_url:
                        MEME_URLS.append(img_url)
                        print(f'[meme] Got meme URL: {img_url}')
                    else:
                        print('[meme] No image URL found in response')
        except Exception as e:
            print(f'[meme] Error: {e}')
        await asyncio.sleep(60)

async def describe_image_with_grok(image_url: str) -> str:
    api_url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-2-vision-latest",
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant created by xAI."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                {"type": "text", "text": "Describe the contents of this image. Use only what you see and do not provide any external info"}
            ]}
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                # The structure of the response may vary; adjust as needed
                try:
                    return result["choices"][0]["message"]["content"]
                except Exception:
                    return str(result)
            else:
                return f"[grok] Error: {resp.status} {await resp.text()}"

async def boston_wrongs_conversation_loop():
    global CURRENT_DESCRIPTION
    # Create the agents
    danny = Agent(
        name="Danny",
        instructions=PROMPT_BASE + DANNY_PERSONA,
        model=LitellmModel(model=model, api_key=GROK_API_KEY),
        model_settings=ModelSettings(temperature=1.5),
    )
    anthony = Agent(
        name="Anthony",
        instructions=PROMPT_BASE + ANTHONY_PERSONA,
        model=LitellmModel(model=model, api_key=GROK_API_KEY),
        model_settings=ModelSettings(temperature=1.5),
    )
    meixi = Agent(
        name="Meixi",
        instructions=PROMPT_BASE + MEIXI_PERSONA,
        model=LitellmModel(model=model, api_key=GROK_API_KEY),
        model_settings=ModelSettings(temperature=1.5),
    )
    agents = [danny, anthony, meixi]
    manager = Agent(
        name="Manager",
        instructions="You are the silent manager of The Boston Wrongs. When given a meme description, you randomly pick one of the group to start the conversation. You never speak yourself.",
        model=LitellmModel(model=model, api_key=GROK_API_KEY),
    )
    last_meme_url = None
    conversation = []
    model_settings = ModelSettings(temperature=1.5)
    while True:
        # Wait for a new meme URL
        while not MEME_URLS or (MEME_URLS[-1] == last_meme_url):
            await asyncio.sleep(1)
        meme_url = MEME_URLS[-1]
        last_meme_url = meme_url
        # Get description
        CURRENT_DESCRIPTION = await describe_image_with_grok(meme_url)
        print(f'\n[NEW MEME] {meme_url}\n[DESCRIPTION] {CURRENT_DESCRIPTION}\n')
        # Manager picks who starts
        first_agent = random.choice(agents)
        others = [a for a in agents if a != first_agent]
        # Start the conversation
        msg = f"Here's a meme: {CURRENT_DESCRIPTION} What do you think?"
        conversation = [
            {"role": "user", "content": msg}
        ]
        # First agent responds
        result = await Runner.run(first_agent, msg)
        print(f"{first_agent.name}: {result.final_output}")
        conversation.append({"role": first_agent.name, "content": result.final_output})
        # The other two agents respond in order
        for agent in others:
            prev = conversation[-1]["content"]
            msg = f"{prev}"
            result = await Runner.run(agent, msg)
            print(f"{agent.name}: {result.final_output}")
            conversation.append({"role": agent.name, "content": result.final_output})
        # Wait for the next meme
        print("\n--- Waiting for next meme ---\n")
        while MEME_URLS[-1] == last_meme_url:
            await asyncio.sleep(1)

async def main():
    meme_task = asyncio.create_task(fetch_meme_url())
    await boston_wrongs_conversation_loop()
    meme_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 