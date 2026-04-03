import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import InsuranceClaimEnvironment
from environment.grader import AgentGrader
from inference.baseline_agent import LLMBaselineAgent
from tqdm import tqdm
import json

def evaluate_agent(env, agent, grader, num_episodes=10, difficulty=None):
    """Evaluate agent across multiple episodes"""
    
    results = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating episodes"):
        # Reset environment
        obs = env.reset(difficulty=difficulty)
        actions = []
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(obs)
            actions.append(action)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
        
        # Grade episode
        episode_result = grader.grade_episode(
            actions,
            info["ground_truth"],
            info["step"],
            env.max_steps
        )
        
        episode_result["scenario_id"] = info["scenario_id"]
        episode_result["difficulty"] = info["difficulty"]
        episode_result["tag"] = info["tag"]
        results.append(episode_result)
    
    # Get summary metrics
    summary = grader.get_summary_metrics()
    
    return results, summary

def main():
    # Initialize environment and agent
    env = InsuranceClaimEnvironment({"max_steps": 6})
    agent = LLMBaselineAgent()
    grader = AgentGrader()
    
    # Evaluate across difficulties
    all_results = {}
    
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n{'='*50}")
        print(f"Evaluating on {difficulty.upper()} scenarios")
        print(f"{'='*50}")
        
        grader.reset()
        results, summary = evaluate_agent(
            env, agent, grader, 
            num_episodes=5, 
            difficulty=difficulty
        )
        
        all_results[difficulty] = {
            "results": results,
            "summary": summary
        }
        
        print(f"\nSummary for {difficulty}:")
        for metric, value in summary.items():
            print(f"  {metric}: {value:.3f}")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Overall Performance")
    print(f"{'='*50}")
    
    # Calculate overall metrics
    overall_metrics = {}
    for difficulty, data in all_results.items():
        for metric, value in data["summary"].items():
            if metric not in overall_metrics:
                overall_metrics[metric] = []
            overall_metrics[metric].append(value)
    
    for metric, values in overall_metrics.items():
        avg_value = sum(values) / len(values)
        print(f"  Average {metric}: {avg_value:.3f}")

if __name__ == "__main__":
    main()