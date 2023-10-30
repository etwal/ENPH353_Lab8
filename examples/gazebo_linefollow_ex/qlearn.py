import random
import pickle
import csv



class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        # TODO: Implement loading Q values from pickle file.

        # Opening the file in binary read mode ("rb") and ensures that the file is properly closed when you're done with it.
        with open(filename + ".pickle", "rb") as file:
            self.q = pickle.load(file)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        
        # Saving to pickle
        with open(filename + ".pickle", "wb") as file:
            pickle.dump(self.q, file)
        print("Wrote to file: {}".format(filename+".pickle"))

        # Saving to CSV
        with open(filename + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["State-Action", "Value"])  # Writing headers
            for key, value in self.q.items():
                writer.writerow([key, value])
        print("Wrote to file: {}".format(filename+".csv"))


        

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        # Exploration: epsilon chance of taking a random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
            if return_q:
                return action, self.getQ(state, action)
            else:
                return action

        # Exploitation: get action with max Q value
        q_values = {action: self.getQ(state, action) for action in self.actions}
        max_q_value = max(q_values.values())

        # Get a list of all actions that have the maximum Q value
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q_value]

        # If multiple actions have the max Q value, randomly choose among them
        chosen_action = random.choice(best_actions)

        if return_q:
            return chosen_action, max_q_value
        else:
            return chosen_action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        # Ensure state-action pair is in the Q-table.
        if (state1, action1) not in self.q:
            self.q[(state1, action1)] = 0  # Initialize Q-value to 0 if not present

        # Calculate max Q-value for state2 over all possible actions.
        # If there's no known action for state2 in the Q-table, the max Q-value is 0.
        max_q_s2 = max([self.q.get((state2, a), 0) for a in self.actions])

        # Update Q-value for (state1, action1) using the Bellman equation.
        self.q[(state1, action1)] += self.alpha * (reward + self.gamma * max_q_s2 - self.q[(state1, action1)])
