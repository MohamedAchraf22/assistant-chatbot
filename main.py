from rag import ask_question


def main():

    print("RAG Chatbot")
    print("Type 'exit' to quit")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()