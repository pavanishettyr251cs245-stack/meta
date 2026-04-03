import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env2 import MyEnv2Action, MyEnv2Env

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = LOCAL_IMAGE_NAME
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("INSURANCE_CLAIM_TASK", "insurance-claim-validation")
BENCHMARK = os.getenv("INSURANCE_CLAIM_BENCHMARK", "insurance-claim-validation")
MAX_STEPS = 6
TEMPERATURE = 0.1
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert insurance claim validator.
    You will be given details of an insurance claim including policy info,
    user history, documents, and risk signals.
    Your job is to analyze the claim and decide one of:
    approve_claim, reject_claim, escalate_claim, request_additional_info,
    analyze_claim, detect_fraud_signals, ignore.
    Reply with exactly one action string and nothing else.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs_dict: dict) -> str:
    claim = obs_dict.get("claim", {})
    policy = obs_dict.get("policy", {})
    user = obs_dict.get("user_history", {})
    docs = obs_dict.get("documents", {})
    violations = obs_dict.get("policy_violations", [])
    signals = obs_dict.get("risk_signals", [])

    doc_lines = "\n".join(
        f"  - {k}: {v.get('status', 'unknown')}" for k, v in docs.items()
    )
    signal_lines = "\n".join(
        f"  - {s.get('signal_type')}: {s.get('description')} "
        f"(severity {s.get('severity', 0):.2f})"
        for s in signals
    ) or "  None"
    violation_lines = "\n".join(f"  - {v}" for v in violations) or "  None"

    return textwrap.dedent(f"""
        Claim Type: {claim.get('claim_type')}
        Amount: ${claim.get('amount')}
        Description: {claim.get('description')}
        Severity: {claim.get('severity')}

        Policy Coverage Limit: ${policy.get('coverage_limits', {}).get(claim.get('claim_type', ''), 'N/A')}
        Excluded Conditions: {', '.join(policy.get('excluded_conditions', [])) or 'None'}
        Required Documents: {', '.join(policy.get('required_documents', []))}

        User Total Claims: {user.get('total_claims')}
        User Risk Score: {user.get('risk_score')}
        Claim Frequency: {user.get('claim_frequency')} per year
        Previously Flagged: {user.get('flagged_previous')}

        Documents:
{doc_lines}

        Policy Violations:
{violation_lines}

        Risk Signals:
{signal_lines}

        Choose exactly one action:
        approve_claim, reject_claim, escalate_claim,
        request_additional_info, analyze_claim, detect_fraud_signals, ignore
    """).strip()


def get_model_action(client: OpenAI, obs_dict: dict) -> str:
    user_prompt = build_prompt(obs_dict)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().lower()
        valid_actions = [
            "approve_claim", "reject_claim", "escalate_claim",
            "request_additional_info", "analyze_claim",
            "detect_fraud_signals", "ignore"
        ]
        for action in valid_actions:
            if action in text:
                return action
        return "analyze_claim"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "analyze_claim"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await MyEnv2Env.from_docker_image(IMAGE_NAME)
    else:
        env = MyEnv2Env(base_url=os.getenv(
            "ENV_URL", "https://sanjan-m-my-env2.hf.space"))

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env as e:
            result = await e.reset()
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_dict = obs.dict() if hasattr(obs, "dict") else {}
                action_str = get_model_action(client, obs_dict)

                from environment.schemas import ClaimAction, ReasoningOutput
                action = MyEnv2Action(
                    action=action_str,
                    reasoning=ReasoningOutput(confidence=0.5),
                    parameters={}
                )

                result = await e.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=error)

                if done:
                    break

        max_total_reward = MAX_STEPS
        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())