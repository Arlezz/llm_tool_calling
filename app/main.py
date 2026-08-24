from app.agent import new_conversation, run_tool_calling_loop

messages = new_conversation()

print("Ctrl+C o /salir para terminar. /reset limpia el contexto.\n")

while True:
    try:
        user_message = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("\nChau!")
        break

    if user_message == "/salir":
        break
    if user_message == "/reset":
        messages = new_conversation()
        print("Contexto reiniciado.\n")
        continue

    messages.append({"role": "user", "content": user_message})
    run_tool_calling_loop(messages, user_message)
