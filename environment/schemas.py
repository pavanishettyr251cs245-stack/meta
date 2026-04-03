from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid

class ClaimType(str, Enum):
    AUTO = "auto"
    HOME = "home"
    HEALTH = "health"
    LIFE = "life"

class DocumentStatus(str, Enum):
    MISSING = "missing"
    PENDING = "pending"
    UPLOADED = "uploaded"
    VERIFIED = "verified"
    REJECTED = "rejected"

class RiskSignal(BaseModel):
    signal_type: str
    description: str
    severity: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class PolicyRule(BaseModel):
    rule_id: str
    rule_type: Literal["coverage_limit", "document_requirement", "waiting_period", "exclusion"]
    condition: str
    requirement: Any
    applies_to: List[ClaimType]
    
class ClaimDetails(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_type: ClaimType
    amount: float = Field(gt=0)
    description: str
    incident_date: datetime
    filing_date: datetime = Field(default_factory=datetime.now)
    location: Optional[str] = None
    severity: Literal["low", "medium", "high"] = "medium"

class UserHistory(BaseModel):
    user_id: str
    total_claims: int = 0
    total_payout: float = 0.0
    previous_claims: List[Dict] = []
    account_age_days: int = Field(default=0)
    claim_frequency: float = Field(default=0.0)  # claims per year
    flagged_previous: bool = False
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)

class Document(BaseModel):
    doc_type: str
    status: DocumentStatus
    url: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    notes: Optional[str] = None

class PolicyInfo(BaseModel):
    policy_id: str
    coverage_limits: Dict[str, float]  # per claim type limit
    deductibles: Dict[str, float]
    waiting_period_days: int = 0
    excluded_conditions: List[str] = []
    required_documents: List[str] = []
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)

class ReasoningOutput(BaseModel):
    policy_violation: bool = False
    claim_amount_valid: bool = True
    user_risk_high: bool = False
    documents_complete: bool = False
    fraud_indicators: List[str] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation: Optional[str] = None
    
    class Config:
        extra = "forbid"

class ClaimObservation(BaseModel):
    claim: ClaimDetails
    policy: PolicyInfo
    user_history: UserHistory
    documents: Dict[str, Document]
    risk_signals: List[RiskSignal] = []
    derived_signals: Dict[str, Any] = {}
    policy_violations: List[str] = []
    step_count: int = 0
    metadata: Dict[str, Any] = {}
    
class ClaimAction(BaseModel):
    action: Literal[
        "analyze_claim",
        "detect_fraud_signals", 
        "approve_claim",
        "reject_claim",
        "escalate_claim",
        "request_additional_info",
        "ignore"
    ]
    reasoning: ReasoningOutput
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"

class RewardInfo(BaseModel):
    score: float
    components: Dict[str, float]
    step_penalty: float = 0.0
    bonus: float = 0.0
    details: Dict[str, Any] = {}