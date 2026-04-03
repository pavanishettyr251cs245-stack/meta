import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import InsuranceClaimEnvironment
from environment.grader import AgentGrader
from inference.baseline_agent import LLMBaselineAgent
import json
import pandas as pd
from datetime import datetime

class InsuranceClaimUI:
    def __init__(self):
        self.env = InsuranceClaimEnvironment({"max_steps": 6})
        self.grader = AgentGrader()
        self.agent = LLMBaselineAgent()
        self.current_episode = None
    
    def process_claim(self, difficulty, scenario_id):
        """Process a claim and return results"""
        
        # Reset environment
        obs = self.env.reset(scenario_id=scenario_id if scenario_id != "random" else None,
                            difficulty=difficulty if difficulty != "all" else None)
        
        actions = []
        done = False
        
        # Run agent
        while not done:
            action = self.agent.get_action(obs)
            actions.append(action)
            obs, reward, done, info = self.env.step(action)
        
        # Grade episode
        result = self.grader.grade_episode(
            actions,
            info["ground_truth"],
            info["step"],
            self.env.max_steps
        )
        
        # Format output
        output = {
            "scenario_info": {
                "id": info["scenario_id"],
                "difficulty": info["difficulty"],
                "tag": info["tag"]
            },
            "claim_details": {
                "type": obs.claim.claim_type.value,
                "amount": f"${obs.claim.amount:,.2f}",
                "description": obs.claim.description
            },
            "agent_decision": {
                "action": actions[-1].action,
                "reasoning": actions[-1].reasoning.dict(),
                "confidence": actions[-1].reasoning.confidence
            },
            "evaluation": {
                "final_score": f"{result['final_score']:.3f}",
                "decision_accuracy": f"{result['decision_accuracy']:.3f}",
                "fraud_detection": f"{result['fraud_detection_quality']:.3f}",
                "reasoning_quality": f"{result['reasoning_quality']:.3f}",
                "action_efficiency": f"{result['action_efficiency']:.3f}",
                "total_steps": result['total_steps']
            },
            "ground_truth": info["ground_truth"]
        }
        
        return output, json.dumps(output, indent=2)
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Insurance Claim Validation System", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🏢 Insurance Claim Validation System")
            gr.Markdown("AI-powered decision engine for fraud detection and claim validation")
            
            with gr.Row():
                with gr.Column(scale=1):
                    difficulty = gr.Dropdown(
                        choices=["easy", "medium", "hard", "all"],
                        label="Difficulty Level",
                        value="medium"
                    )
                    scenario_id = gr.Textbox(
                        label="Scenario ID (optional, leave 'random' for random)",
                        value="random"
                    )
                    process_btn = gr.Button("Process Claim", variant="primary")
                
                with gr.Column(scale=2):
                    output_json = gr.JSON(label="Results")
                    output_text = gr.Textbox(label="Raw Output", lines=10)
            
            process_btn.click(
                fn=self.process_claim,
                inputs=[difficulty, scenario_id],
                outputs=[output_json, output_text]
            )
            
            gr.Markdown("""
            ### System Overview
            
            This environment simulates an insurance claim validation system with:
            
            - **3 Difficulty Levels**: Easy, Medium, Hard scenarios
            - **Policy Rule Engine**: Deterministic validation against coverage limits, exclusions, and requirements
            - **Fraud Detection**: Pattern-based fraud indicators including:
              - Claims near policy limits
              - High claim frequency
              - Suspicious filing patterns
              - First-time high-value claims
            - **Multi-factor Reasoning**: Evaluates policy violations, amount validity, user risk, and document completeness
            - **Structured Grading**: 0.0-1.0 score based on decision accuracy, fraud detection, reasoning quality, and efficiency
            
            ### Actions Available
            - `analyze_claim`: Examine claim details
            - `detect_fraud_signals`: Identify suspicious patterns
            - `approve_claim`: Accept valid claim
            - `reject_claim`: Deny invalid claim
            - `escalate_claim`: Flag for manual review
            - `request_additional_info`: Get missing documents
            - `ignore`: Skip (penalized)
            
            ### Grading Weights
            - Decision Accuracy: 40%
            - Fraud Detection Quality: 30%
            - Reasoning Quality: 20%
            - Action Efficiency: 10%
            """)
        
        return demo

if __name__ == "__main__":
    ui = InsuranceClaimUI()
    demo = ui.create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)