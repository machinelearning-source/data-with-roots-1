"""Q-Learning engine (Reinforcement Learning) for the Data with Roots project.

The agent walks a fixed 10x10 grid looking for the target cell while avoiding
walls and dangerous zones. The Q-function is approximated with a scikit-learn
``SGDRegressor`` trained online through ``partial_fit``, using a one-hot
encoding of the (state, action) pair. The tabular update

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

is therefore *learned* from experience instead of being hard-coded, and every
number reported by the application comes from that training run.
"""

import numpy as np
from sklearn.linear_model import SGDRegressor

ROWS = 10
COLS = 10
N_ACTIONS = 4

MAZE = (
    "Aooooooooo",
    "oDDooooooo",
    "#oDDoo##oo",
    "#ooDDoo#oo",
    "o#ooDDo#oo",
    "##oooDD#oo",
    "oo##ooo##o",
    "o#o###oooo",
    "o#o#oooooo",
    "oooooooooT",
)

ACTIONS = ("Up", "Down", "Left", "Right")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))

CELL_AGENT = "A"
CELL_TARGET = "T"
CELL_WALL = "#"
CELL_DANGER = "D"
CELL_PATH = "o"

REWARD_STEP = -1.0
REWARD_WALL = -10.0
REWARD_DANGER = -15.0
REWARD_GOAL = 100.0

GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98
EPISODES = 300
MAX_STEPS = 30
LEARNING_RATE = 0.8
SEED = 42

CELL_LABELS = {
    CELL_AGENT: "Agent (Start)",
    CELL_TARGET: "Target (Goal)",
    CELL_PATH: "Path",
    CELL_WALL: "Wall",
    CELL_DANGER: "Danger Zone",
}

EXPECTED_COUNTS = {
    "agent": 1,
    "target": 1,
    "wall": 20,
    "danger": 10,
    "path": 68,
    "total": 100,
}

REWARD_TABLE = (
    ("Valid move to a normal cell", REWARD_STEP),
    ("Move blocked by a wall or the grid limit", REWARD_WALL),
    ("Entered a danger zone", REWARD_DANGER),
    ("Reached the target cell", REWARD_GOAL),
)

ENVIRONMENT_TABLE = (
    (CELL_AGENT, "Agent start position", "Every episode begins here."),
    (CELL_TARGET, "Target (goal)", "Terminates the episode with a positive reward."),
    (CELL_PATH, "Free path cell", "A normal, safe move."),
    (CELL_WALL, "Wall", "Cannot be entered; the agent stays in place."),
    (CELL_DANGER, "Danger zone", "Can be entered but is heavily penalised."),
)


def maze():
    """Return the maze as a list of rows (list of single-character strings)."""
    return [list(row) for row in MAZE]


def in_grid(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS


def _find(char):
    for row in range(ROWS):
        for col in range(COLS):
            if MAZE[row][col] == char:
                return (row, col)
    raise RuntimeError("Cell %r is missing from the maze" % char)


def start_state():
    return _find(CELL_AGENT)


def target_state():
    return _find(CELL_TARGET)


def counts():
    total = {}
    for row in MAZE:
        for char in row:
            total[char] = total.get(char, 0) + 1
    return {
        "agent": total.get(CELL_AGENT, 0),
        "target": total.get(CELL_TARGET, 0),
        "wall": total.get(CELL_WALL, 0),
        "danger": total.get(CELL_DANGER, 0),
        "path": total.get(CELL_PATH, 0),
        "total": sum(total.values()),
    }


def validate():
    """Return a list of human readable problems with the maze definition."""
    data = counts()
    problems = []
    for key, want in EXPECTED_COUNTS.items():
        if data[key] != want:
            problems.append("%s: found %s, expected %s" % (key, data[key], want))
    for row in MAZE:
        if len(row) != COLS:
            problems.append("row %r has %d columns" % (row, len(row)))
    return problems


def step(state, action):
    """Apply ``action`` to ``state``.

    Returns ``(next_state, reward, done, cell_name)``. Blocked moves keep the
    agent in place and penalise it instead of ending the episode, so the agent
    has to *learn* that walls are not worth trying.
    """
    row = state[0] + DELTAS[action][0]
    col = state[1] + DELTAS[action][1]
    if not in_grid(row, col):
        return state, REWARD_WALL, False, "Outside the grid"
    char = MAZE[row][col]
    if char == CELL_WALL:
        return state, REWARD_WALL, False, "Wall"
    if char == CELL_TARGET:
        return (row, col), REWARD_GOAL, True, "Target"
    if char == CELL_DANGER:
        return (row, col), REWARD_DANGER, False, "Danger Zone"
    return (row, col), REWARD_STEP, False, "Path"


def _state_offset(state):
    return state[0] * COLS * N_ACTIONS + state[1] * N_ACTIONS


def q_values(model, state):
    """Return the four Q-values of ``state`` in a single predict call."""
    features = np.zeros((N_ACTIONS, ROWS * COLS * N_ACTIONS))
    offsets = _state_offset(state) + np.arange(N_ACTIONS)
    features[np.arange(N_ACTIONS), offsets] = 1.0
    return model.predict(features)


def _apply(model, state, action, target):
    features = np.zeros((1, ROWS * COLS * N_ACTIONS))
    features[0, _state_offset(state) + action] = 1.0
    model.partial_fit(features, np.array([target]))


def _greedy(model, state, rng):
    """Pick the best action, breaking ties at random.

    Ties are common while the Q-values are still zero, so always taking the
    lowest index would make the agent bounce between two cells instead of
    exploring.
    """
    values = q_values(model, state)
    best = values.max()
    candidates = np.flatnonzero(values >= best - 1e-12)
    return int(rng.choice(candidates))


def train(episodes=EPISODES, gamma=GAMMA, epsilon_start=EPSILON_START,
          epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
          max_steps=MAX_STEPS, learning_rate=LEARNING_RATE, seed=SEED):
    """Run epsilon-greedy Q-Learning and return the trained model plus stats."""
    rng = np.random.default_rng(seed)
    # fit_intercept=False is required: with one active feature per sample the
    # shared intercept would accumulate every residual and shift every
    # prediction, which stops the Q-values from converging.
    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=learning_rate,
        random_state=seed,
    )
    _apply(model, (0, 0), 0, 0.0)

    origin = start_state()
    goal = target_state()
    epsilon = epsilon_start
    successes = 0
    total_reward = 0.0
    rewards = []
    epsilons = []

    for _ in range(episodes):
        state = origin
        episode_reward = 0.0
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = _greedy(model, state, rng)
            next_state, reward, done, _ = step(state, action)
            if done:
                target = reward
            else:
                target = reward + gamma * float(np.max(q_values(model, next_state)))
            _apply(model, state, action, target)
            state = next_state
            episode_reward += reward
            if done:
                if state == goal:
                    successes += 1
                break
        total_reward += episode_reward
        rewards.append(episode_reward)
        epsilons.append(epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return {
        "model": model,
        "episodes": episodes,
        "successes": successes,
        "success_rate": 100.0 * successes / episodes if episodes else 0.0,
        "average_reward": total_reward / episodes if episodes else 0.0,
        "final_epsilon": epsilon,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "rewards": rewards,
        "epsilons": epsilons,
    }


def evaluate(model, max_steps=MAX_STEPS, seed=SEED):
    """Run the greedy policy (epsilon = 0) and trace every movement."""
    rng = np.random.default_rng(seed)
    origin = start_state()
    goal = target_state()
    state = origin
    rows = []
    path = [state]
    total = 0.0
    for index in range(1, max_steps + 1):
        action = _greedy(model, state, rng)
        next_state, reward, done, cell = step(state, action)
        rows.append({
            "step": index,
            "state": state,
            "action": ACTIONS[action],
            "next_state": next_state,
            "cell": cell,
            "reward": reward,
        })
        total += reward
        state = next_state
        path.append(state)
        if done:
            break
    return {
        "steps": rows,
        "path": path,
        "total_reward": total,
        "movements": len(rows),
        "goal_reached": state == goal,
        "danger_hits": sum(1 for row in rows if row["cell"] == "Danger Zone"),
        "blocked_moves": sum(1 for row in rows if row["reward"] == REWARD_WALL),
    }


def q_table(model):
    """Return the learned Q-values for every reachable (non-wall) state."""
    rows = []
    for row in range(ROWS):
        for col in range(COLS):
            char = MAZE[row][col]
            if char == CELL_WALL:
                continue
            values = q_values(model, (row, col))
            rows.append({
                "state": (row, col),
                "cell": char,
                "label": CELL_LABELS[char],
                "up": float(values[0]),
                "down": float(values[1]),
                "left": float(values[2]),
                "right": float(values[3]),
            })
    return rows


def render_grid(path_states=()):
    """Return the grid for the template, flagging the cells on the path."""
    marked = set(path_states)
    grid = []
    for row in range(ROWS):
        cells = []
        for col in range(COLS):
            char = MAZE[row][col]
            cells.append({
                "char": char,
                "label": CELL_LABELS[char],
                "on_path": (row, col) in marked,
            })
        grid.append(cells)
    return grid
