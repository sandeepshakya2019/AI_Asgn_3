import numpy as np

class RandomAgent:
    """
    A simple agent that selects random actions.
    The action space consists of 3 binary values (0 or 1).
    """
    def act(self, observation):
        """
        Generate a random action consisting of 3 binary values.
        
        Args:
            observation: The current state of the environment (unused in this random agent).

        Returns:
            A numpy array of size 3 with random binary values (0 or 1).
        """
        return np.random.randint(2, size=3)