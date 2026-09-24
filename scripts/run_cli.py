from src.agent.graph import ask_agent

def main():
    print("Travel Memory Agent CLI")
    print("Type 'exit' to stop.\n")
    history = ""

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = ask_agent(question, history=history)
        print("\nAgent:\n", result["answer"])
        print("\n", result["verification"], "\n")

        history += f"\nUser: {question}\nAssistant: {result['answer']}\n"

if __name__ == "__main__":
    main()
