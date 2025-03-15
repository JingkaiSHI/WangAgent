import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = None

    def select_child(self, c=1.4):
        """
        select child using Upper Confidence Bound formula (UCB)
        """
        return max(self.children.values(), 
                   key=lambda child: child.value / (child.visits + 1e-5) + 
                   c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5)))
    
    def is_fully_expanded(self):
        """
        check if all possible actions have been tried at least once
        """
        return self.untried_actions is None or len(self.untried_actions) == 0
    
    def expand(self, action, next_state):
        """
        add a new child node for the given action and next state
        """
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child_node
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child_node
    
    def update(self, result):
        """
        update the node with the result of a simulation
        """
        self.visits += 1
        self.value += result

    def get_best_action(self):
        """
        return the action with the highest visit count
        """
        if not self.children:
            return None
        return max(self.children.items(), key=lambda item: item[1].visits)[0]
    