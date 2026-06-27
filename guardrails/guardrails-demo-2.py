import os

from dotenv import load_dotenv

from openai import OpenAI
from guardrails import Guard

from guardrails.validators import register_validator, PassResult, FailResult

load_dotenv()

client = OpenAI(
    base_url="https://opencode.ai/zen/v1/", api_key=os.getenv("LLM_API_KEY")
)


def zen_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


@register_validator(name="simple_topic_check", data_type="string")
def simple_topic_check(value, metadata):
    financial_keyword = ["stock", "apple", "investment", "ticker", "finance", "market"]

    if any(keyword in value.lower() for keyword in financial_keyword):
        return PassResult()
    else:
        return FailResult(errorMessage="Query is not about financial topics")


guard = Guard().use(simple_topic_check(on_fail="exception"))
queries = [
    "How is Apple stock doing?",
    "What's the weather today",
]

for query in queries:
    print(f"\nQuery: {query}")

    try:
        guard.validate(query)
        result = zen_wrapper(messages=[{"role": "user", "content": query}])
        print(result)

    except Exception as e:
        print(f"BLOCKED: {e}")
