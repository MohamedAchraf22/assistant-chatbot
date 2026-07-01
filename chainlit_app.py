import chainlit as cl
import httpx
from conversation_memory import save_conversation   
FASTAPI_URL = "http://localhost:8000/chat"


@cl.on_chat_start
async def start():
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

    try:
        print(f"Calling FastAPI → {FASTAPI_URL}")
        
        async with httpx.AsyncClient(timeout=180.0) as client:  
            response = await client.post(
                FASTAPI_URL,
                json={"question": user_input},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_reply = data.get("answer", "No answer received.")
                
                # Streaming
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
    try:
        save_conversation(user_input, bot_reply)
    except Exception as e:
        print(f"Failed to save conversation: {e}")

    print("=== FINISHED ===")