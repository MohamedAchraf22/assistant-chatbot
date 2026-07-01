from conversation_memory import clear_conversation_history, get_conversation_vectorstore
import os

def rebuild_conversation_store():
    print(" Clearing all previous conversation history...")
    
    success = clear_conversation_history()
    
    if success:
        # Creating new vector store
        vectorstore = get_conversation_vectorstore()
        print(" New conversation vector store created successfully!")
        print(f" Location: {os.path.join('chroma', 'conversations')}")
    else:
        print(" Something went wrong while clearing history.")


if __name__ == "__main__":
    print("="*50)
    print("   Conversation Memory Ingest Tool")
    print("="*50)
    print("This will DELETE ALL previous conversations.")
    print()
    rebuild_conversation_store()