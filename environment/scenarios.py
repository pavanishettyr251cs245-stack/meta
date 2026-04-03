from typing import Dict, Any, List
from datetime import datetime, timedelta
from .schemas import (
    ClaimType, DocumentStatus, ClaimDetails, PolicyInfo, 
    UserHistory, Document, RiskSignal
)

class ScenarioGenerator:
    """Generate realistic claim scenarios with ground truth"""
    
    def __init__(self):
        self.scenarios = self._build_scenarios()
    
    def _build_scenarios(self) -> List[Dict[str, Any]]:
        """Build all scenarios with ground truth"""
        
        scenarios = []
        
        # Easy Scenarios
        scenarios.extend([
            {
                "id": "easy_001",
                "difficulty": "easy",
                "tag": "clean",
                "description": "Valid auto claim within limits, complete docs",
                "claim": {
                    "type": ClaimType.AUTO,
                    "amount": 2500,
                    "description": "Rear-ended at traffic light, moderate damage to bumper",
                    "incident_date": datetime.now() - timedelta(days=3),
                    "severity": "medium"
                },
                "policy": {
                    "coverage_limits": {"auto": 5000},
                    "deductibles": {"auto": 500},
                    "waiting_period_days": 0,
                    "excluded_conditions": [],
                    "required_documents": ["police_report", "photos", "estimate"]
                },
                "user": {
                    "total_claims": 1,
                    "total_payout": 1800,
                    "previous_claims": [{"date": datetime.now() - timedelta(days=365), "amount": 1800}],
                    "account_age_days": 730,
                    "claim_frequency": 0.5,
                    "flagged_previous": False,
                    "risk_score": 0.2
                },
                "documents": {
                    "police_report": DocumentStatus.VERIFIED,
                    "photos": DocumentStatus.VERIFIED,
                    "estimate": DocumentStatus.VERIFIED
                },
                "ground_truth": {
                    "correct_action": "approve_claim",
                    "fraud_label": False,
                    "has_policy_violation": False,
                    "amount_valid": True,
                    "user_high_risk": False,
                    "docs_complete": True
                }
            },
            {
                "id": "easy_002",
                "difficulty": "easy",
                "tag": "policy_violation",
                "description": "Claim exceeding coverage limits",
                "claim": {
                    "type": ClaimType.HOME,
                    "amount": 15000,
                    "description": "Water damage from burst pipe, flooding basement",
                    "incident_date": datetime.now() - timedelta(days=5),
                    "severity": "high"
                },
                "policy": {
                    "coverage_limits": {"home": 10000},
                    "deductibles": {"home": 1000},
                    "waiting_period_days": 0,
                    "excluded_conditions": ["flood"],
                    "required_documents": ["photos", "contractor_estimate", "plumber_report"]
                },
                "user": {
                    "total_claims": 0,
                    "total_payout": 0,
                    "previous_claims": [],
                    "account_age_days": 180,
                    "claim_frequency": 0,
                    "flagged_previous": False,
                    "risk_score": 0.1
                },
                "documents": {
                    "photos": DocumentStatus.VERIFIED,
                    "contractor_estimate": DocumentStatus.VERIFIED,
                    "plumber_report": DocumentStatus.VERIFIED
                },
                "ground_truth": {
                    "correct_action": "reject_claim",
                    "fraud_label": False,
                    "has_policy_violation": True,
                    "amount_valid": False,
                    "user_high_risk": False,
                    "docs_complete": True
                }
            }
        ])
        
        # Medium Scenarios
        scenarios.extend([
            {
                "id": "medium_001",
                "difficulty": "medium",
                "tag": "borderline",
                "description": "Claim at policy limit with missing docs",
                "claim": {
                    "type": ClaimType.HEALTH,
                    "amount": 5000,
                    "description": "Emergency room visit for severe allergic reaction",
                    "incident_date": datetime.now() - timedelta(days=2),
                    "severity": "high"
                },
                "policy": {
                    "coverage_limits": {"health": 5000},
                    "deductibles": {"health": 250},
                    "waiting_period_days": 0,
                    "excluded_conditions": [],
                    "required_documents": ["medical_report", "hospital_bill", "prescription"]
                },
                "user": {
                    "total_claims": 2,
                    "total_payout": 3200,
                    "previous_claims": [
                        {"date": datetime.now() - timedelta(days=200), "amount": 1500},
                        {"date": datetime.now() - timedelta(days=400), "amount": 1700}
                    ],
                    "account_age_days": 500,
                    "claim_frequency": 1.46,
                    "flagged_previous": False,
                    "risk_score": 0.4
                },
                "documents": {
                    "medical_report": DocumentStatus.UPLOADED,
                    "hospital_bill": DocumentStatus.PENDING,
                    "prescription": DocumentStatus.MISSING
                },
                "ground_truth": {
                    "correct_action": "request_additional_info",
                    "fraud_label": False,
                    "has_policy_violation": False,
                    "amount_valid": True,
                    "user_high_risk": False,
                    "docs_complete": False
                }
            },
            {
                "id": "medium_002",
                "difficulty": "medium",
                "tag": "fraud_suspicion",
                "description": "Suspicious claim with inconsistent details",
                "claim": {
                    "type": ClaimType.AUTO,
                    "amount": 4800,
                    "description": "Hit and run, car damaged",
                    "incident_date": datetime.now() - timedelta(days=1),
                    "severity": "medium"
                },
                "policy": {
                    "coverage_limits": {"auto": 5000},
                    "deductibles": {"auto": 500},
                    "waiting_period_days": 0,
                    "excluded_conditions": [],
                    "required_documents": ["police_report", "photos", "estimate"]
                },
                "user": {
                    "total_claims": 3,
                    "total_payout": 12500,
                    "previous_claims": [
                        {"date": datetime.now() - timedelta(days=30), "amount": 4500},
                        {"date": datetime.now() - timedelta(days=90), "amount": 3800},
                        {"date": datetime.now() - timedelta(days=150), "amount": 4200}
                    ],
                    "account_age_days": 200,
                    "claim_frequency": 5.475,
                    "flagged_previous": True,
                    "risk_score": 0.7
                },
                "documents": {
                    "police_report": DocumentStatus.UPLOADED,
                    "photos": DocumentStatus.UPLOADED,
                    "estimate": DocumentStatus.PENDING
                },
                "ground_truth": {
                    "correct_action": "escalate_claim",
                    "fraud_label": True,
                    "has_policy_violation": False,
                    "amount_valid": True,
                    "user_high_risk": True,
                    "docs_complete": False
                }
            }
        ])
        
        # Hard Scenarios
        scenarios.extend([
            {
                "id": "hard_001",
                "difficulty": "hard",
                "tag": "conflicting_signals",
                "description": "Complex claim with mixed signals",
                "claim": {
                    "type": ClaimType.HOME,
                    "amount": 9500,
                    "description": "Fire damage from electrical malfunction. Previous claim 6 months ago for similar issue.",
                    "incident_date": datetime.now() - timedelta(days=4),
                    "severity": "high"
                },
                "policy": {
                    "coverage_limits": {"home": 10000},
                    "deductibles": {"home": 1000},
                    "waiting_period_days": 0,
                    "excluded_conditions": ["electrical_fire"],
                    "required_documents": ["fire_report", "photos", "contractor_estimate", "electrical_inspection"]
                },
                "user": {
                    "total_claims": 2,
                    "total_payout": 8800,
                    "previous_claims": [
                        {"date": datetime.now() - timedelta(days=180), "amount": 8800, "type": "electrical_fire"}
                    ],
                    "account_age_days": 365,
                    "claim_frequency": 2.0,
                    "flagged_previous": True,
                    "risk_score": 0.6
                },
                "documents": {
                    "fire_report": DocumentStatus.VERIFIED,
                    "photos": DocumentStatus.VERIFIED,
                    "contractor_estimate": DocumentStatus.UPLOADED,
                    "electrical_inspection": DocumentStatus.MISSING
                },
                "ground_truth": {
                    "correct_action": "escalate_claim",
                    "fraud_label": True,
                    "has_policy_violation": True,
                    "amount_valid": True,
                    "user_high_risk": True,
                    "docs_complete": False
                }
            },
            {
                "id": "hard_002",
                "difficulty": "hard",
                "tag": "sophisticated_fraud",
                "description": "Sophisticated fraud with professional documentation",
                "claim": {
                    "type": ClaimType.AUTO,
                    "amount": 4950,
                    "description": "Multi-vehicle collision on highway, extensive damage to front and side. All documentation provided.",
                    "incident_date": datetime.now() - timedelta(days=10),
                    "severity": "high"
                },
                "policy": {
                    "coverage_limits": {"auto": 5000},
                    "deductibles": {"auto": 500},
                    "waiting_period_days": 0,
                    "excluded_conditions": [],
                    "required_documents": ["police_report", "photos", "estimate", "witness_statements"]
                },
                "user": {
                    "total_claims": 0,
                    "total_payout": 0,
                    "previous_claims": [],
                    "account_age_days": 25,
                    "claim_frequency": 0,
                    "flagged_previous": False,
                    "risk_score": 0.1
                },
                "documents": {
                    "police_report": DocumentStatus.VERIFIED,
                    "photos": DocumentStatus.VERIFIED,
                    "estimate": DocumentStatus.VERIFIED,
                    "witness_statements": DocumentStatus.VERIFIED
                },
                "ground_truth": {
                    "correct_action": "reject_claim",
                    "fraud_label": True,
                    "has_policy_violation": False,
                    "amount_valid": True,
                    "user_high_risk": True,
                    "docs_complete": True
                }
            }
        ])
        
        return scenarios
    
    def get_scenario(self, scenario_id: str = None, difficulty: str = None) -> Dict[str, Any]:
        """Get a specific scenario or random one by difficulty"""
        
        if scenario_id:
            scenario = next((s for s in self.scenarios if s["id"] == scenario_id), None)
            if scenario:
                return scenario
        
        if difficulty:
            filtered = [s for s in self.scenarios if s["difficulty"] == difficulty]
            if filtered:
                import random
                return random.choice(filtered)
        
        import random
        return random.choice(self.scenarios)