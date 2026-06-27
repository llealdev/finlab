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


@register_validator(name="zen_topic_check", data_type="string")
def zen_topic_check(value, metadata):
    prompt = f"""Is this query about financial analysis or stocks? Answer YES or NO. Query: {value}"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    if "YES" in response.choices[0].message.content.upper():
        return PassResult()
    else:
        return FailResult(errorMessage="Not about finance")


guard = Guard().use(zen_topic_check(on_fail="exception"))
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
