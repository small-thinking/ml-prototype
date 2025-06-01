import numpy as np


q_table = {}
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate


def get_q(state, action):
    return q_table.get((tuple(state), action), 0.0)


def update_q(state, action, reward, next_state, done):
    state_key = tuple(state)
    if state_key not in q_table:
        q_table[state_key] = np.zeros(4)

    if done and reward < 0:
        new_value = reward
    else:
        next_max_q = max(get_q(next_state, a) for a in range(4))
        new_value = get_q(state, action) + alpha * (
            reward + gamma * next_max_q - get_q(state, action)
        )

    q_table[state_key][action] = new_value


def select_action(state):
    dist = state[3]
    direction = state[4]

    if np.random.rand() < epsilon:
        if dist < 2:
            return 0 if direction > 0 else 1
        elif dist == 2:
            return 2
        else:
            return np.random.randint(3)

    q_values = [get_q(state, a) for a in range(4)]

    if dist < 2:
        if direction > 0:
            q_values[0] += 1
        else:
            q_values[1] += 1

    q_values[3] -= 1

    return int(np.argmax(q_values))
