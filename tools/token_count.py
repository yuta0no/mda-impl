import anthropic
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    client = anthropic.Anthropic()

    response = client.messages.count_tokens(
        model="claude-opus-5-5",
        system="You are a scientist",
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )

    print(response.model_dump_json())
