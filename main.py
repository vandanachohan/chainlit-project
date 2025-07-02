import os
from dotenv import load_dotenv
import chainlit as cl
from typing import cast

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set in .env")

from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner
from agents.run import RunConfig

@cl.on_chat_start
async def start():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/",
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

    await cl.Message(content="👋 Welcome to Giaic AI Assistant! Ask anything .").send()


@cl.on_message
async def handle_message(message: cl.Message):
    msg = cl.Message(content="⏳ Processing your question...")
    await msg.send()

    agent = cast(Agent, cl.user_session.get("agent"))
    config = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat history")
    history.append({"role": "user", "content": message.content})

    file_texts = cl.user_session.get("file_texts", [])
    file_qa_context = "\n\n".join([f"{name}:\n{content[:3000]}" for name, content in file_texts])

    try:
        if file_texts:
            prompt = f"You are an expert assistant. Use the following files to answer the question:\n\n{file_qa_context}\n\nQuestion: {message.content}"
            result = Runner.run_sync(
                starting_agent=agent,
                input=[{"role": "user", "content": prompt}],
                run_config=config
            )
        else:
            result = Runner.run_sync(
                starting_agent=agent,
                input=history,
                run_config=config
            )

        response_content = result.final_output
        print(f"🧠 Assistant: {response_content}")

        msg.content = response_content
        await msg.update()

        cl.user_session.set("chat history", result.to_input_list())

    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()
        print(f"❌ Error: {str(e)}")
