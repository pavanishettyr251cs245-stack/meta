from typing import Dict, Any, List
from .schemas import ClaimAction, ReasoningOutput
import numpy as np

class AgentGrader:
    """Deterministic grader for agent performance"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_results = []
        self.total_score = 0.0
        self.metrics = {
            "decision_accuracy": [],
            "fraud_detection": [],
            "reasoning_quality": [],
            "action_efficiency": []
        }
    
    def grade_episode(self, actions: List[ClaimAction], 
                     ground_truth: Dict[str, Any],
                     total_steps: int,
                     max_steps: int) -> Dict[str, Any]:
        """Grade a complete episode"""
        
        # Find terminal action
        terminal_actions = ["approve_claim", "reject_claim", "escalate_claim"]
        terminal_action = None
        for action in actions:
            if action.action in terminal_actions:
                terminal_action = action
                break
        
        if not terminal_action:
            terminal_action = actions[-1]
        
        # Decision accuracy
        decision_correct = terminal_action.action == ground_truth["correct_action"]
        decision_score = 1.0 if decision_correct else 0.0
        
        # Fraud detection quality
        fraud_detected = len(terminal_action.reasoning.fraud_indicators) > 0
        actual_fraud = ground_truth["fraud_label"]
        
        if actual_fraud and fraud_detected:
            fraud_score = 1.0
        elif not actual_fraud and not fraud_detected:
            fraud_score = 1.0
        elif actual_fraud and not fraud_detected:
            fraud_score = 0.0
        else:
            fraud_score = 0.5
        
        # Reasoning quality
        reasoning_factors = [
            terminal_action.reasoning.policy_violation == ground_truth.get("has_policy_violation", False),
            terminal_action.reasoning.claim_amount_valid == ground_truth.get("amount_valid", True),
            terminal_action.reasoning.user_risk_high == ground_truth.get("user_high_risk", False),
            terminal_action.reasoning.documents_complete == ground_truth.get("docs_complete", False)
        ]
        reasoning_score = sum(reasoning_factors) / len(reasoning_factors)
        
        # Action efficiency
        efficiency_score = max(0, 1 - (total_steps / max_steps))
        
        # Penalties for lazy strategies
        penalty = 0.0
        if all(a.action == "escalate_claim" for a in actions):
            penalty -= 0.3  # Always escalating
        elif all(a.action == "reject_claim" for a in actions):
            penalty -= 0.3  # Always rejecting
        elif len([a for a in actions if a.action == "ignore"]) > 2:
            penalty -= 0.2  # Too many ignore actions
        
        # Final score
        final_score = (
            0.4 * decision_score +
            0.3 * fraud_score +
            0.2 * reasoning_score +
            0.1 * efficiency_score +
            penalty
        )
        final_score = np.clip(final_score, 0.0, 1.0)
        
        # Store metrics
        self.metrics["decision_accuracy"].append(decision_score)
        self.metrics["fraud_detection"].append(fraud_score)
        self.metrics["reasoning_quality"].append(reasoning_score)
        self.metrics["action_efficiency"].append(efficiency_score)
        
        result = {
            "final_score": final_score,
            "decision_accuracy": decision_score,
            "fraud_detection_quality": fraud_score,
            "reasoning_quality": reasoning_score,
            "action_efficiency": efficiency_score,
            "total_steps": total_steps,
            "terminal_action": terminal_action.action,
            "correct_decision": decision_correct,
            "fraud_detected": fraud_detected,
            "penalty_applied": penalty != 0,
            "penalty_amount": penalty
        }
        
        self.episode_results.append(result)
        return result
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across episodes"""
        
        if not self.episode_results:
            return {}
        
        return {
            "avg_final_score": np.mean([r["final_score"] for r in self.episode_results]),
            "avg_decision_accuracy": np.mean(self.metrics["decision_accuracy"]),
            "avg_fraud_detection": np.mean(self.metrics["fraud_detection"]),
            "avg_reasoning_quality": np.mean(self.metrics["reasoning_quality"]),
            "avg_action_efficiency": np.mean(self.metrics["action_efficiency"]),
            "std_final_score": np.std([r["final_score"] for r in self.episode_results]),
            "total_episodes": len(self.episode_results),
            "fraud_detection_rate": np.mean([1 if r["fraud_detected"] else 0 
                                            for r in self.episode_results]),
            "false_positive_rate": np.mean([1 if r["fraud_detected"] and not 
                                           ground_truth.get("fraud_label", False) 
                                           else 0 for r in self.episode_results]),
            "avg_steps_per_episode": np.mean([r["total_steps"] for r in self.episode_results])
        }