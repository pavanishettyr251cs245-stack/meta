import os
import json
from typing import Dict, Any
from openai import OpenAI
from environment.schemas import ClaimAction, ReasoningOutput, ClaimObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

class LLMBaselineAgent:
    """Baseline agent using LLM for claim validation"""
    
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
    def get_action(self, observation: ClaimObservation) -> ClaimAction:
        """Generate action based on observation"""
        
        prompt = self._build_prompt(observation)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an insurance claim validator. Analyze claims and provide structured decisions. Respond only in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            text = response.choices[0].message.content or ""
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            result = json.loads(text)
            
            reasoning = ReasoningOutput(
                policy_violation=result.get("policy_violation", False),
                claim_amount_valid=result.get("claim_amount_valid", True),
                user_risk_high=result.get("user_risk_high", False),
                documents_complete=result.get("documents_complete", False),
                fraud_indicators=result.get("fraud_indicators", []),
                confidence=result.get("confidence", 0.5),
                recommendation=result.get("recommended_action")
            )
            
            action = ClaimAction(
                action=result.get("action", "analyze_claim"),
                reasoning=reasoning,
                parameters=result.get("parameters", {})
            )
            
            return action
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return ClaimAction(
                action="analyze_claim",
                reasoning=ReasoningOutput(confidence=0.5),
                parameters={}
            )
    
    def _build_prompt(self, obs: ClaimObservation) -> str:
        return f"""
Claim Type: {obs.claim.claim_type}
Amount: ${obs.claim.amount}
Description: {obs.claim.description}
Severity: {obs.claim.severity}

Policy Coverage Limit: ${obs.policy.coverage_limits.get(obs.claim.claim_type.value, 0)}
Excluded Conditions: {', '.join(obs.policy.excluded_conditions) or 'None'}
Required Documents: {', '.join(obs.policy.required_documents)}

User Total Claims: {obs.user_history.total_claims}
User Risk Score: {obs.user_history.risk_score}
Claim Frequency: {obs.user_history.claim_frequency} per year
Previously Flagged: {obs.user_history.flagged_previous}

Documents: {', '.join(f"{k}: {v.status}" for k, v in obs.documents.items())}
Policy Violations: {', '.join(obs.policy_violations) or 'None'}

Respond in JSON:
{{
    "action": "one of: analyze_claim, detect_fraud_signals, approve_claim, reject_claim, escalate_claim, request_additional_info, ignore",
    "policy_violation": true/false,
    "claim_amount_valid": true/false,
    "user_risk_high": true/false,
    "documents_complete": true/false,
    "fraud_indicators": [],
    "confidence": 0.0-1.0,
    "recommended_action": "final decision",
    "parameters": {{}}
}}
"""
    
    def _format_documents(self, documents: Dict) -> str:
        return '\n'.join(f"- {k}: {v.status}" for k, v in documents.items())
    
    def _format_risk_signals(self, signals: list) -> str:
        if not signals:
            return "No risk signals"
        return '\