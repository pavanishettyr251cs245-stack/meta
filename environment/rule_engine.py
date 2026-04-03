from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from .schemas import (
    ClaimObservation, ClaimDetails, PolicyInfo, UserHistory,
    Document, DocumentStatus, RiskSignal, ReasoningOutput
)

class PolicyRuleEngine:
    """Deterministic rule engine for policy validation and fraud detection"""
    
    def __init__(self):
        self.rule_weights = {
            "policy_violation": 0.3,
            "claim_amount": 0.2,
            "document_completeness": 0.2,
            "user_history": 0.15,
            "fraud_patterns": 0.15
        }
    
    def evaluate_claim(self, observation: ClaimObservation) -> Dict[str, Any]:
        """Evaluate claim against all rules and return structured assessment"""
        
        results = {
            "policy_violations": self._check_policy_violations(observation),
            "amount_validity": self._validate_claim_amount(observation),
            "document_status": self._check_documents(observation),
            "user_risk": self._assess_user_risk(observation),
            "fraud_indicators": self._detect_fraud_patterns(observation),
            "recommended_action": None,
            "confidence": 0.0
        }
        
        # Generate recommended action based on rules
        if results["policy_violations"]:
            results["recommended_action"] = "reject_claim"
            results["confidence"] = 0.9
        elif results["fraud_indicators"] and len(results["fraud_indicators"]) >= 2:
            results["recommended_action"] = "escalate_claim"
            results["confidence"] = 0.85
        elif not results["document_status"]["complete"]:
            results["recommended_action"] = "request_additional_info"
            results["confidence"] = 0.8
        elif results["user_risk"]["risk_score"] > 0.7:
            results["recommended_action"] = "escalate_claim"
            results["confidence"] = 0.75
        else:
            results["recommended_action"] = "approve_claim"
            results["confidence"] = self._calculate_confidence(results)
        
        return results
    
    def _check_policy_violations(self, obs: ClaimObservation) -> List[str]:
        """Check for policy violations"""
        violations = []
        
        # Check coverage limits
        limit = obs.policy.coverage_limits.get(obs.claim.claim_type.value, 0)
        if obs.claim.amount > limit:
            violations.append(f"Claim amount ${obs.claim.amount} exceeds coverage limit ${limit}")
        
        # Check waiting period
        days_since_policy = (obs.claim.incident_date - obs.policy.created_at).days
        if days_since_policy < obs.policy.waiting_period_days:
            violations.append(f"Incident occurred during waiting period ({days_since_policy} days)")
        
        # Check exclusions
        for exclusion in obs.policy.excluded_conditions:
            if exclusion.lower() in obs.claim.description.lower():
                violations.append(f"Claim involves excluded condition: {exclusion}")
        
        # Check if policy is active
        if not obs.policy.active:
            violations.append("Policy is not active")
        
        return violations
    
    def _validate_claim_amount(self, obs: ClaimObservation) -> Dict[str, Any]:
        """Validate claim amount reasonableness"""
        
        avg_claim_amounts = {
            "auto": 5000,
            "home": 10000,
            "health": 3000,
            "life": 50000
        }
        
        avg = avg_claim_amounts.get(obs.claim.claim_type.value, 5000)
        ratio = obs.claim.amount / avg
        
        return {
            "valid": ratio <= 3.0,  # Within 3x average
            "ratio": ratio,
            "is_high_value": ratio > 2.0,
            "is_low_value": ratio < 0.5
        }
    
    def _check_documents(self, obs: ClaimObservation) -> Dict[str, Any]:
        """Check document completeness and validity"""
        
        required = set(obs.policy.required_documents)
        uploaded = {doc_type for doc_type, doc in obs.documents.items() 
                   if doc.status in [DocumentStatus.UPLOADED, DocumentStatus.VERIFIED]}
        
        missing = required - uploaded
        pending = {doc_type for doc_type, doc in obs.documents.items() 
                  if doc.status == DocumentStatus.PENDING}
        
        # Check for invalid/rejected documents
        rejected = {doc_type for doc_type, doc in obs.documents.items() 
                   if doc.status == DocumentStatus.REJECTED}
        
        return {
            "complete": len(missing) == 0 and len(rejected) == 0,
            "missing": list(missing),
            "pending": list(pending),
            "rejected": list(rejected),
            "total_required": len(required),
            "total_uploaded": len(uploaded)
        }
    
    def _assess_user_risk(self, obs: ClaimObservation) -> Dict[str, Any]:
        """Assess user risk based on history"""
        
        risk_score = obs.user_history.risk_score
        
        # Update risk based on claim frequency
        if obs.user_history.claim_frequency > 2.0:  # More than 2 claims per year
            risk_score = min(1.0, risk_score + 0.2)
        
        # Flag for recent claims
        recent_claims = [c for c in obs.user_history.previous_claims 
                        if (datetime.now() - c.get("date", datetime.now())).days < 180]
        if len(recent_claims) > 2:
            risk_score = min(1.0, risk_score + 0.15)
        
        # Check for flagged history
        if obs.user_history.flagged_previous:
            risk_score = min(1.0, risk_score + 0.3)
        
        return {
            "risk_score": risk_score,
            "total_claims": obs.user_history.total_claims,
            "total_payout": obs.user_history.total_payout,
            "claim_frequency": obs.user_history.claim_frequency,
            "is_high_risk": risk_score > 0.7,
            "new_user": obs.user_history.account_age_days < 30
        }
    
    def _detect_fraud_patterns(self, obs: ClaimObservation) -> List[Dict[str, Any]]:
        """Detect potential fraud indicators"""
        
        indicators = []
        
        # Pattern 1: Amount just below limit
        limit = obs.policy.coverage_limits.get(obs.claim.claim_type.value, 0)
        if limit > 0 and obs.claim.amount > limit * 0.95:
            indicators.append({
                "type": "amount_near_limit",
                "severity": 0.6,
                "description": f"Claim amount ${obs.claim.amount} is within 5% of limit ${limit}"
            })
        
        # Pattern 2: Multiple claims in short period
        if obs.user_history.claim_frequency > 3.0:
            indicators.append({
                "type": "high_claim_frequency",
                "severity": 0.8,
                "description": f"Claim frequency {obs.user_history.claim_frequency:.1f} claims/year"
            })
        
        # Pattern 3: Inconsistent dates
        days_diff = (obs.claim.filing_date - obs.claim.incident_date).days
        if days_diff < 1:
            indicators.append({
                "type": "immediate_filing",
                "severity": 0.5,
                "description": "Claim filed on same day as incident"
            })
        elif days_diff > 90:
            indicators.append({
                "type": "delayed_filing",
                "severity": 0.4,
                "description": f"Claim filed {days_diff} days after incident"
            })
        
        # Pattern 4: High value for first-time claimant
        if obs.user_history.total_claims == 0 and obs.claim.amount > 10000:
            indicators.append({
                "type": "high_value_first_claim",
                "severity": 0.7,
                "description": f"First-time claimant requesting ${obs.claim.amount}"
            })
        
        # Pattern 5: Vague description
        if len(obs.claim.description.split()) < 10:
            indicators.append({
                "type": "vague_description",
                "severity": 0.3,
                "description": "Claim description is unusually brief"
            })
        
        return indicators
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence in recommendation"""
        
        # Base confidence
        confidence = 0.7
        
        # Adjust based on conflicting signals
        conflict_count = 0
        
        if results["policy_violations"] and not results["fraud_indicators"]:
            conflict_count += 1
        if results["user_risk"]["risk_score"] > 0.5 and not results["fraud_indicators"]:
            conflict_count += 1
        if not results["document_status"]["complete"] and results["policy_violations"]:
            conflict_count += 1
        
        confidence -= conflict_count * 0.1
        
        return max(0.5, min(0.95, confidence))
    
    def compute_reward(self, agent_action: ReasoningOutput, 
                      ground_truth: Dict[str, Any],
                      step_count: int,
                      max_steps: int = 6) -> Dict[str, float]:
        """Compute reward based on agent's reasoning and ground truth"""
        
        # Decision accuracy (0.4 weight)
        decision_correct = agent_action.recommendation == ground_truth["correct_action"]
        decision_score = 1.0 if decision_correct else 0.0
        
        # Fraud detection quality (0.3 weight)
        fraud_detected = len(agent_action.fraud_indicators) > 0
        actual_fraud = ground_truth["fraud_label"]
        
        if actual_fraud and fraud_detected:
            fraud_score = 1.0
        elif not actual_fraud and not fraud_detected:
            fraud_score = 1.0
        elif actual_fraud and not fraud_detected:
            fraud_score = 0.0  # Missed fraud
        else:
            fraud_score = 0.5  # False positive
        
        # Reasoning quality (0.2 weight)
        reasoning_score = self._evaluate_reasoning(agent_action, ground_truth)
        
        # Action efficiency (0.1 weight)
        efficiency_score = max(0, 1 - (step_count / max_steps))
        
        # Step penalty for repeated actions or inefficiency
        step_penalty = 0.0
        
        # Combine scores
        total_score = (
            0.4 * decision_score +
            0.3 * fraud_score +
            0.2 * reasoning_score +
            0.1 * efficiency_score -
            step_penalty
        )
        
        return {
            "total": total_score,
            "decision": decision_score,
            "fraud_detection": fraud_score,
            "reasoning": reasoning_score,
            "efficiency": efficiency_score
        }
    
    def _evaluate_reasoning(self, agent_reasoning: ReasoningOutput,
                           ground_truth: Dict[str, Any]) -> float:
        """Evaluate reasoning quality against ground truth factors"""
        
        factors = [
            ("policy_violation", agent_reasoning.policy_violation, 
             ground_truth.get("has_policy_violation", False)),
            ("claim_amount_valid", agent_reasoning.claim_amount_valid,
             ground_truth.get("amount_valid", True)),
            ("user_risk_high", agent_reasoning.user_risk_high,
             ground_truth.get("user_high_risk", False)),
            ("documents_complete", agent_reasoning.documents_complete,
             ground_truth.get("docs_complete", False))
        ]
        
        correct_count = sum(1 for _, agent, truth in factors if agent == truth)
        return correct_count / len(factors)