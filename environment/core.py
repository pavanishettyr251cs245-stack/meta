from typing import Dict, Any, Tuple, Optional
import numpy as np
from .schemas import (
    ClaimObservation, ClaimAction, RewardInfo, ClaimDetails,
    PolicyInfo, UserHistory, Document, DocumentStatus, RiskSignal,
    ReasoningOutput
)
from .rule_engine import PolicyRuleEngine
from .scenarios import ScenarioGenerator
import copy

class InsuranceClaimEnvironment:
    """OpenEnv-compliant Insurance Claim Validation Environment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rule_engine = PolicyRuleEngine()
        self.scenario_gen = ScenarioGenerator()
        self.current_scenario = None
        self.current_observation = None
        self.step_count = 0
        self.max_steps = self.config.get("max_steps", 6)
        self.action_history = []
        self.done = False
        
    def reset(self, scenario_id: str = None, difficulty: str = None) -> ClaimObservation:
        """Reset environment with new claim scenario"""
        
        # Load scenario
        self.current_scenario = self.scenario_gen.get_scenario(scenario_id, difficulty)
        self.step_count = 0
        self.action_history = []
        self.done = False
        
        # Build observation
        self.current_observation = self._build_observation()
        
        return copy.deepcopy(self.current_observation)
    
    def _build_observation(self) -> ClaimObservation:
        """Build observation from current scenario"""
        
        scenario = self.current_scenario
        
        # Create claim details
        claim = ClaimDetails(**scenario["claim"])
        
        # Create policy info
        policy = PolicyInfo(
            policy_id=f"POL_{scenario['id']}",
            coverage_limits=scenario["policy"]["coverage_limits"],
            deductibles=scenario["policy"]["deductibles"],
            waiting_period_days=scenario["policy"]["waiting_period_days"],
            excluded_conditions=scenario["policy"]["excluded_conditions"],
            required_documents=scenario["policy"]["required_documents"],
            active=True
        )
        
        # Create user history
        user = UserHistory(
            user_id=f"USER_{scenario['id']}",
            **scenario["user"]
        )
        
        # Create documents
        documents = {}
        for doc_type, status in scenario["documents"].items():
            documents[doc_type] = Document(
                doc_type=doc_type,
                status=status
            )
        
        # Initial risk signals (if any from scenario)
        risk_signals = []
        if "risk_signals" in scenario:
            risk_signals = [RiskSignal(**sig) for sig in scenario["risk_signals"]]
        
        # Evaluate policy violations
        temp_obs = ClaimObservation(
            claim=claim,
            policy=policy,
            user_history=user,
            documents=documents,
            risk_signals=risk_signals,
            step_count=0
        )
        
        policy_violations = self.rule_engine._check_policy_violations(temp_obs)
        
        # Create observation
        observation = ClaimObservation(
            claim=claim,
            policy=policy,
            user_history=user,
            documents=documents,
            risk_signals=risk_signals,
            derived_signals={},
            policy_violations=policy_violations,
            step_count=0,
            metadata={
                "scenario_id": scenario["id"],
                "difficulty": scenario["difficulty"],
                "tag": scenario["tag"]
            }
        )
        
        return observation
    
    def step(self, action: ClaimAction) -> Tuple[ClaimObservation, float, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done, info"""
        
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() first.")
        
        # Validate action
        self._validate_action(action)
        
        # Record action
        self.action_history.append(action)
        self.step_count += 1
        
        # Process action
        reward_info = self._process_action(action)
        
        # Check if episode should end
        terminal_actions = ["approve_claim", "reject_claim", "escalate_claim"]
        if action.action in terminal_actions or self.step_count >= self.max_steps:
            self.done = True
        
        # Update observation
        self.current_observation.step_count = self.step_count
        self.current_observation.derived_signals["last_action"] = action.action
        
        # Update risk signals based on action history
        self._update_risk_signals()
        
        # Prepare info
        info = {
            "scenario_id": self.current_scenario["id"],
            "difficulty": self.current_scenario["difficulty"],
            "tag": self.current_scenario["tag"],
            "step": self.step_count,
            "reward_components": reward_info.components,
            "ground_truth": self.current_scenario["ground_truth"]
        }
        
        return (
            copy.deepcopy(self.current_observation),
            reward_info.score,
            self.done,
            info
        )
    
    def state(self) -> Dict[str, Any]:
        """Return current state representation"""
        
        return {
            "observation": self.current_observation.dict(),
            "step_count": self.step_count,
            "action_history": [a.dict() for a in self.action_history],
            "done": self.done,
            "scenario_metadata": self.current_scenario.get("metadata", {})
        }
    
    def _validate_action(self, action: ClaimAction):
        """Validate action format and reasoning"""
        
        # Basic validation
        assert action.action in ["analyze_claim", "detect_fraud_signals", "approve_claim", 
                                 "reject_claim", "escalate_claim", "request_additional_info", "ignore"]
        
        # Validate reasoning fields
        assert isinstance(action.reasoning, ReasoningOutput)
    
    def _process_action(self, action: ClaimAction) -> RewardInfo:
        """Process action and compute reward"""
        
        ground_truth = self.current_scenario["ground_truth"]
        
        # Update observation based on action
        if action.action == "request_additional_info":
            # Simulate document upload for next step
            self._add_documents()
        elif action.action == "detect_fraud_signals":
            # Add fraud detection result to signals
            self._add_fraud_signals(action.reasoning)
        elif action.action == "analyze_claim":
            # Update derived signals based on reasoning
            self._update_derived_signals(action.reasoning)
        
        # Compute reward using rule engine
        reward_components = self.rule_engine.compute_reward(
            action.reasoning,
            ground_truth,
            self.step_count,
            self.max_steps
        )
        
        # Add penalty for inefficient actions
        if action.action == "ignore":
            reward_components["total"] -= 0.1
        elif len(self.action_history) > 1 and self.action_history[-2].action == action.action:
            # Penalty for repeated same action
            reward_components["total"] -= 0.05
        
        # Bonus for correct terminal action at appropriate time
        terminal_actions = ["approve_claim", "reject_claim", "escalate_claim"]
        if action.action in terminal_actions:
            if action.action == ground_truth["correct_action"]:
                reward_components["total"] += 0.1
            else:
                reward_components["total"] -= 0.15
        
        reward_components["total"] = np.clip(reward_components["total"], -1.0, 1.0)
        
        return RewardInfo(
            score=reward_components["total"],
            components=reward_components,
            step_penalty=0.0,
            bonus=0.0
        )
    
    def _add_documents(self):
        """Simulate document addition"""
        
        # Add missing documents
        required = set(self.current_observation.policy.required_documents)
        for doc_type in required:
            if doc_type not in self.current_observation.documents:
                self.current_observation.documents[doc_type] = Document(
                    doc_type=doc_type,
                    status=DocumentStatus.UPLOADED
                )
            elif self.current_observation.documents[doc_type].status == DocumentStatus.MISSING:
                self.current_observation.documents[doc_type].status = DocumentStatus.UPLOADED
    
    def _add_fraud_signals(self, reasoning: ReasoningOutput):
        """Add fraud signals from agent reasoning"""
        
        for indicator in reasoning.fraud_indicators:
            self.current_observation.risk_signals.append(
                RiskSignal(
                    signal_type="agent_detected_fraud",
                    description=indicator,
                    severity=0.7
                )
            )
    
    def _update_derived_signals(self, reasoning: ReasoningOutput):
        """Update derived signals based on agent reasoning"""
        
        self.current_observation.derived_signals["agent_reasoning"] = reasoning.dict()
    
    def _update_risk_signals(self):
        """Update risk signals based on action history"""
        
        # Add signal for indecisive behavior
        if len(self.action_history) >= 3:
            recent_actions = [a.action for a in self.action_history[-3:]]
            if all(a == "analyze_claim" for a in recent_actions):
                self.current_observation.risk_signals.append(
                    RiskSignal(
                        signal_type="indecisive_behavior",
                        description="Agent repeatedly analyzing without decision",
                        severity=0.5
                    )
                )