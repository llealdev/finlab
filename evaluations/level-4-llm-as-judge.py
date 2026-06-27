import os
from typing import Literal

from dotenv import load_dotenv
from langfuse import Langfuse
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


load_dotenv()

langfuse = Langfuse()

client = OpenAI(
    base_url="https://opencode.ai/zen/v1/", api_key=os.getenv("LLM_API_KEY")
)
client = instructor.from_openai(client, mode=instructor.Mode.JSON)


class FundamentalAnalysisEvaluation(BaseModel):
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1")
    reasoning: str = Field(description="Explanation of the score")
    pass_fail: Literal["pass", "fail"] = Field(description="Binary pass/fail")


FUNDAMENTAL_EVAL_PROMPT = """ You are a expert financial analyst evaluating the quality of a fundamental analysis.

INPUT (User Query): 
{input}:

OUTPUT (Analysis Generated):
Investment Grade: {investiment_grade}
Confidence Score: {confidence_score}
Key Strengths: {key_strengths}
Key Concerns: {key_concerns}
Recommendation: {recommendation}

Evaluate this fundamental analysis based on:
1. Does the investment grade seem reasonable? 
2. Are the key strengths actually strengths for this company?
3. Are the key concerns relevant and important?
4. Is the recommendation logical given the analysis?
5. Does the confidence score make sense?

Provide: 
- score: 0.0 to 1.0 (1.0 = excellent analysis, 0.0 = poor analysis)
- reasoning: Brief explanation of your evaluation
- pass_fail: "pass" if score >=0.7, otherwise "fail"
"""


def evaluate_fundamental_analysis(trace_data: dict) -> FundamentalAnalysisEvaluation:
    input_query = trace_data.get("input", {})
    output_data = trace_data.get("output", {})
    fundamental = output_data.get("fundamental_analysis", {})

    prompt = FUNDAMENTAL_EVAL_PROMPT.format(
        input=input_query.get("query", ""),
        investiment_grade=fundamental.get("investment_grade", ""),
        confidence_score=fundamental.get("confidence_score", ""),
        key_strengths=", ".join(fundamental.get("key_strengths", [])),
        key_concerns=", ".join(fundamental.get("key_concerns", [])),
        recommendation=fundamental.get("recommendation", ""),
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=FundamentalAnalysisEvaluation,
    )

    return response


def evaluate_recommendation_quality(trace_data: dict) -> FundamentalAnalysisEvaluation:
    output_data = trace_data.get("output", {})
    final_rec = output_data.get("final_recommendation", {})

    prompt = f"""You are an expert financial analyst evaluating investment recommendations.

    RECOMMENDATION:
    Action: {final_rec.get("action", "")}
    Confidence: {final_rec.get("confidence", "")}
    Rationale: {final_rec.get("rationale", "")}
    Key Risks: {", ".join(final_rec.get("key_risks", []))}
    Key Opportunities: {", ".join(final_rec.get("key_opportunities", []))}

    Evaluate this recommendation based on:
    1. Is the action (BUY/HOLD/SELL) clear and decisive?
    2. Does the confidence level match the strength of rationale?
    3. Does the rationale provide concrete reasoning?
    4. Are key risks and opportunities relevant and specific?

    Provide:
    - score: 0.0 to 1.0 (1.0 = excellent recommendation, 0.0 = poor recommendation)
    - reasoning: Brief explanation of your evaluation
    - pass_fail: "pass" if score >= 0.7, otherwise "fail"
    """

    response = client.chat.completions.create(
        model="deepseek-v4-flash-free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_model=FundamentalAnalysisEvaluation,
    )

    return response


def run_llm_as_judge_evaluations():
    traces = langfuse.api.trace.list(tags=["evaluation", "integration_test"], limit=1)

    for trace in traces.data:
        trace_data = langfuse.api.trace.get(trace_id=trace.id)

        if not trace_data.output:
            continue

        fundamental_eval = evaluate_fundamental_analysis(
            {"input": trace_data.input, "output": trace_data.output}
        )

        langfuse.create_score(
            trace_id=trace.id,
            name="fundamental_quality",
            value=fundamental_eval.score,
            comment=fundamental_eval.reasoning,
            data_type="NUMERIC",
        )

        recommendation_eval = evaluate_recommendation_quality(
            {"output": trace_data.output}
        )

        langfuse.create_score(
            trace_id=trace.id,
            name="recommendation_quality",
            value=recommendation_eval.score,
            comment=recommendation_eval.reasoning,
            data_type="NUMERIC",
        )

        print(
            f"{trace.name}: fundamental={fundamental_eval.score:.2f}, recommendation={recommendation_eval.score:.2f}"
        )

    langfuse.flush()


if __name__ == "__main__":
    run_llm_as_judge_evaluations()
