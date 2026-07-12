import chainlit as cl
import httpx
from conversation_memory import save_conversation

FASTAPI_URL = "http://localhost:8000/chat"
SESSION_WINDOW = 6  # number of recent turns to keep per session


@cl.on_chat_start
async def start():
    # Initialise an empty turn list for this session
    cl.user_session.set("history", [])
    print("=== CHAT STARTED ===")
    await cl.Message(
        content="Hi! 👋 I'm ready. Send me your message..."
    ).send()
    print("WELCOME MESSAGE SENT")


@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    print(f"=== MESSAGE RECEIVED ===\nUser: {user_input}")

    msg = cl.Message(content="Thinking...")
    await msg.send()

    bot_reply = "Sorry, something went wrong."

    # Retrieve the current session window (list of {role, content} dicts)
    history: list[dict] = cl.user_session.get("history") or []

    try:
        print(f"Calling FastAPI → {FASTAPI_URL}")

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                FASTAPI_URL,
                json={"question": user_input, "history": history},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                bot_reply = data.get("answer", "No answer received.")

                # Stream the reply token-by-token
                msg.content = ""
                for token in bot_reply:
                    await msg.stream_token(token)
            else:
                bot_reply = f"Server Error: {response.status_code}"
                msg.content = bot_reply
                await msg.update()

    except httpx.ReadTimeout:
        bot_reply = "The request took too long. Please try again."
        msg.content = bot_reply
        await msg.update()
    except Exception as e:
        bot_reply = f"Error: {str(e)}"
        msg.content = bot_reply
        await msg.update()
        print(f"Error: {e}")

    # Append this turn to the session window, keep last SESSION_WINDOW turns
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})
    cl.user_session.set("history", history[-(SESSION_WINDOW * 2):])

    # Persist to long-term Chroma store (unchanged behaviour)
    try:
        save_conversation(user_input, bot_reply)
    except Exception as e:
        print(f"Failed to save conversation: {e}")

    print("=== FINISHED ===")