# reinforcement_rla/rl_rca.py
import numpy as np
from collections import defaultdict

class RCAReinforcementLearner:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(3))  # State -> action values
        self.actions = [
            "check_logs", 
            "restart_service", 
            "rollback_deployment"
        ]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        
        # Pre-defined rewards for different outcomes
        self.rewards = {
            "success": 10,
            "partial_success": 5,
            "no_effect": 0,
            "worsened": -5,
            "catastrophic": -10
        }
    
    def choose_action(self, state: str) -> str:
        """Choose an action based on current state using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            # Explore: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploit: choose the best known action
            action_values = self.q_table[state]
            return self.actions[np.argmax(action_values)]
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value based on the observed reward"""
        action_index = self.actions.index(action)
        old_value = self.q_table[state][action_index]
        
        # Q-learning formula
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        
        self.q_table[state][action_index] = new_value
    
    def get_recommended_actions(self, state: str, top_n: int = 3) -> List[Dict]:
        """Get recommended actions for a given state"""
        if state not in self.q_table:
            return []
        
        action_values = self.q_table[state]
        # Get indices of top N actions
        top_indices = np.argsort(action_values)[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'action': self.actions[idx],
                'confidence': action_values[idx],
                'steps': self._get_action_steps(self.actions[idx])
            })
        
        return recommendations
    
    def _get_action_steps(self, action: str) -> List[str]:
        """Get detailed steps for an action"""
        action_steps = {
            "check_logs": [
                "1. Identify relevant log files based on the issue",
                "2. Search for error patterns and exceptions",
                "3. Correlate timestamps with incident reports",
                "4. Document findings for future reference"
            ],
            "restart_service": [
                "1. Identify the affected service",
                "2. Check if there are any dependencies",
                "3. Execute graceful restart command",
                "4. Verify service health after restart"
            ],
            "rollback_deployment": [
                "1. Identify the recent deployment",
                "2. Check rollback compatibility",
                "3. Execute rollback procedure",
                "4. Verify system stability post-rollback"
            ]
        }
        
        return action_steps.get(action, ["No specific steps available"])