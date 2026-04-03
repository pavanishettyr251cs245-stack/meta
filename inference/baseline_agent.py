import os
import json
from typing import Dict, Any
import openai
from environment.schemas import ClaimAction, ReasoningOutput, ClaimObservation

class LLMBaselineAgent:
    """Baseline agent using LLM for claim validation"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
    def get_action(self, observation: ClaimObservation) -> ClaimAction:
        """Generate action based on observation"""
        
        # Prepare prompt
        prompt = self._build_prompt(observation)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an insurance claim validator. Analyze claims and provide structured decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Create reasoning output
            reasoning = ReasoningOutput(
                policy_violation=result.get("policy_violation", False),
                claim_amount_valid=result.get("claim_amount_valid", True),
                user_risk_high=result.get("user_risk_high", False),
                documents_complete=result.get("documents_complete", False),
                fraud_indicators=result.get("fraud_indicators", []),
                confidence=result.get("confidence", 0.5),
                recommendation=result.get("recommended_action")
            )
            
            # Create action
            action = ClaimAction(
                action=result.get("action", "analyze_claim"),
                reasoning=reasoning,
                parameters=result.get("parameters", {})
            )
            
            return action
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            # Fallback to safe action
            return ClaimAction(
                action="analyze_claim",
                reasoning=ReasoningOutput(confidence=0.5),
                parameters={}
            )
    
    def _build_prompt(self, obs: ClaimObservation) -> str:
        """Build prompt for LLM"""
        
        prompt = f"""
Claim Analysis Task:

Claim Details:
- Type: {obs.claim.claim_type}
- Amount: ${obs.claim.amount}
- Description: {obs.claim.description}
- Incident Date: {obs.claim.incident_date}
- Severity: {obs.claim.severity}

Policy Information:
- Coverage Limit: ${obs.policy.coverage_limits.get(obs.claim.claim_type.value, 0)}
- Deductible: ${obs.policy.deductibles.get(obs.claim.claim_type.value, 0)}
- Required Documents: {', '.join(obs.policy.required_documents)}
- Excluded Conditions: {', '.join(obs.policy.excluded_conditions)}

User History:
- Total Claims: {obs.user_history.total_claims}
- Total Payout: ${obs.user_history.total_payout}
- Account Age: {obs.user_history.account_age_days} days
- Claim Frequency: {obs.user_history.claim_frequency:.2f} claims/year
- Risk Score: {obs.user_history.risk_score:.2f}
- Previously Flagged: {obs.user_history.flagged_previous}

Documents Status:
{self._format_documents(obs.documents)}

Risk Signals Detected:
{self._format_risk_signals(obs.risk_signals)}

Policy Violations:
{', '.join(obs.policy_violations) if obs.policy_violations else 'None'}

Please analyze this claim and provide a structured response in JSON format:

{{
    "action": "one of: analyze_claim, detect_fraud_signals, approve_claim, reject_claim, escalate_claim, request_additional_info, ignore",
    "policy_violation": true/false,
    "claim_amount_valid": true/false,
    "user_risk_high": true/false,
    "documents_complete": true/false,
    "fraud_indicators": ["list", "of", "suspicious", "patterns"],
    "confidence": 0.0-1.0,
    "recommended_action": "recommended final decision",
    "parameters": {{}}
}}

Base your decision on policy rules, document completeness, user history, and fraud indicators.
"""
        
        return prompt
    
    def _format_documents(self, documents: Dict) -> str:
        """Format document status for prompt"""
        
        lines = []
        for doc_type, doc in documents.items():
            lines.append(f"- {doc_type}: {doc.status}")
        return '\n'.join(lines) if lines else "All documents in order"
    
    def _format_risk_signals(self, signals: list) -> str:
        """Format risk signals for prompt"""
        
        if not signals:
            return "No risk signals detected"
        
        lines = []
        for signal in signals:
            lines.append(f"- {signal.signal_type}: {signal.description} (Severity: {signal.severity})")
        return '\n'.join(lines)