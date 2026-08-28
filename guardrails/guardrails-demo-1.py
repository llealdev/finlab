import os

from dotenv import load_dotenv
from guardrails.hub import ProfanityFree
from openai import OpenAI

from guardrails import Guard

load_dotenv()

client = OpenAI(base_url="https://opencode.ai/zen/v1/", api_key=os.getenv("LLM_API_KEY"))


def zen_wrapper(*, messages, **kwargs) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


guard = Guard().use(ProfanityFree(on_fail="exception"))
query = "FAANG representa quais fucking empresas de tecnologias?"

try:
    guard.validate(query)
except Exception as e:
    print("ENTRADA QUE CONTEM PALAVRAO ", e)

validated_response = guard(
    zen_wrapper,
    messages=[{"role": "user", "content": query}],
)

print("SAIDA: ", validated_response.validated_output)
