import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set in .env")

from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner
from agents.run import RunConfig
from typing import cast

@cl.on_chat_start
async def start():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=client
    )

    config = RunConfig(model=model, model_provider=client, tracing_disabled=True)
    cl.user_session.set("config", config)
    cl.user_session.set("chat history", [])

    agent = Agent(name="Assistant", instructions="You are a helpful assistant.", model=model)
    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to Giaic AI Assistant!").send()

@cl.on_message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="Process...")
    await msg.send()

    agent = cast(Agent, cl.user_session.get("agent"))
    config = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat history")
    history.append({"role": "user", "content": message.content})

    try:
        result = Runner.run_sync(starting_agent=agent, input=history, run_config=config)
        response_content = result.final_output

        print(f"Assistant: {response_content}")

        msg.content = response_content  # ✅ Correct field
        await msg.update()              # ✅ Critical step

        cl.user_session.set("chat history", result.to_input_list())

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
