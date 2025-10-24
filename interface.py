import argparse
import sys
import time
from model_loader import load_model
from chat_memory import ChatMemory

sys.stdout.reconfigure(line_buffering=True)

def build_prompt(memory, user_message):
    
    context = memory.get_context()
    instruction = (
        "You are a helpful assistant. Always answer factual questions in complete sentences. "
        "If the user asks for a capital, respond with 'The capital of X is Y.' format."
    )

    if context:
        prompt = f"{instruction}\n{context}\nUser: {user_message}\nBot:"
    else:
        prompt = f"{instruction}\nUser: {user_message}\nBot:"

    return prompt

def main():
    parser = argparse.ArgumentParser(description="Local CLI Chatbot")
    parser.add_argument("--model", type=str, default="google/flan-t5-large", help="Model name")
    parser.add_argument("--window", type=int, default=6, help="Memory window size")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens per reply")
    parser.add_argument("--device", type=int, default=-1, help="Device (-1=CPU, 0=GPU)")

    args = parser.parse_args()

    print("[INFO] Loading model...")
    generator = load_model(args.model, device=args.device)
    memory = ChatMemory(max_turns=args.window)

    print("\n[READY] Chatbot initialized successfully. Type /exit to quit.\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/exit":
                print("Exiting chatbot. Goodbye!")
                break

            memory.add_user(user_input)
            prompt = build_prompt(memory, user_input)

            print("[INFO] Generating response...")
            start = time.time()

            outputs = generator(
                prompt,
                max_new_tokens=args.max_tokens,
                do_sample=False,  
                temperature=0,
                num_return_sequences=1,
            )

            elapsed = time.time() - start
            raw_output = outputs[0]["generated_text"]
            bot_reply = raw_output[len(prompt):].strip() if raw_output.startswith(prompt) else raw_output.strip()

            for stop_token in ["\nUser:", "User:", "Bot:", "\nBot:"]:
                idx = bot_reply.find(stop_token)
                if idx != -1:
                    bot_reply = bot_reply[:idx].strip()
                    break

            if not bot_reply:
                bot_reply = "(No meaningful reply generated)"

            print(f"Bot: {bot_reply}")
            print(f"[INFO] Response generated in {elapsed:.2f}s\n")

            memory.add_bot(bot_reply)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting chatbot. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()