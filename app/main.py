from agent import new_cache, run_tool_calling_loop, tokenizer

cache = new_cache()

print("Ctrl+C o /salir para terminar. /reset limpia el contexto.\n")

# The KV cache carries the conversation state across turns, so we only ever
# need to build/encode the new user message or tool result, never the history.
while True:
    try:
        user_message = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("\nChau!")
        break

    if user_message == "/salir":
        break
    if user_message == "/reset":
        cache = new_cache()
        print("Contexto reiniciado.\n")
        continue

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        add_generation_prompt=True,
        enable_thinking=True,
    )

    run_tool_calling_loop(prompt, cache, user_message)
