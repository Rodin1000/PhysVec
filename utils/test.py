from send_chat_yidong import send_chat_diverse_model

MODELS_TO_TEST = [
    "gemini-2.5-flash-nothinking",
]

if __name__ == "__main__":
    for model in MODELS_TO_TEST:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")
        try:
            text, status, tokens = send_chat_diverse_model(
                user_model=model,
                role_prompt="",
                user_prompt="Hello, this is a connectivity test. write a short story about a cat.",
                temperature=0.2,
                max_tokens=200,
                iscaltoken=True,
                isstream=True,
            )
            print(f"  status : {status}")
            print(f"  tokens : {tokens}")
            print(f"  text   : {text[:200]}")
            if status == 200:
                print(f"  RESULT : OK")
            else:
                print(f"  RESULT : FAILED (status {status})")
        except Exception as e:
            print(f"  RESULT : ERROR - {e}")

