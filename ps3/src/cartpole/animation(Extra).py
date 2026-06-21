
from __future__ import division, print_function
from math import sin, cos, pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import lfilter

# ============================================================
# CART-POLE PHYSICS & ENVIRONMENT
# ============================================================
class Physics:
    gravity = 9.8
    force_mag = 10.0
    tau = 0.02

class CartPole:
    def __init__(self, physics):
        self.physics = physics
        self.mass_cart = 1.0
        self.mass_pole = 0.3
        self.mass = self.mass_cart + self.mass_pole
        self.length = 0.7
        self.pole_mass_length = self.mass_pole * self.length

    def simulate(self, action, state_tuple):
        x, x_dot, theta, theta_dot = state_tuple
        costheta, sintheta = cos(theta), sin(theta)
        force = self.physics.force_mag if action > 0 else (-1 * self.physics.force_mag)
        temp = (force + self.pole_mass_length * theta_dot * theta_dot * sintheta) / self.mass
        theta_acc = (self.physics.gravity * sintheta - temp * costheta) /                     (self.length * (4/3 - self.mass_pole * costheta * costheta / self.mass))
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.mass
        new_x = x + self.physics.tau * x_dot
        new_x_dot = x_dot + self.physics.tau * x_acc
        new_theta = theta + self.physics.tau * theta_dot
        new_theta_dot = theta_dot + self.physics.tau * theta_acc
        return (new_x, new_x_dot, new_theta, new_theta_dot)

    def get_state(self, state_tuple):
        x, x_dot, theta, theta_dot = state_tuple
        one_deg = pi / 180
        six_deg = 6 * pi / 180
        twelve_deg = 12 * pi / 180
        fifty_deg = 50 * pi / 180
        total_states = 163
        state = 0
        if x < -2.4 or x > 2.4 or theta < -twelve_deg or theta > twelve_deg:
            state = total_states - 1
        else:
            if x < -1.5:
                state = 0
            elif x < 1.5:
                state = 1
            else:
                state = 2
            if x_dot < -0.5:
                pass
            elif x_dot < 0.5:
                state += 3
            else:
                state += 6
            if theta < -six_deg:
                pass
            elif theta < -one_deg:
                state += 9
            elif theta < 0:
                state += 18
            elif theta < one_deg:
                state += 27
            elif theta < six_deg:
                state += 36
            else:
                state += 45
            if theta_dot < -fifty_deg:
                pass
            elif theta_dot < fifty_deg:
                state += 54
            else:
                state += 108
        return state

# ============================================================
# MDP LEARNING
# ============================================================
def initialize_mdp_data(num_states):
    mdp = {
        "num_states": num_states,
        "probs": np.full([num_states, 2, num_states], 1.0/num_states),
        "rewards": np.zeros([num_states]),
        "value": np.random.rand(num_states) * 0.1,
        "count_SAS": np.zeros([num_states, 2, num_states]),
        "count_R": np.zeros([num_states, 2])
    }
    return mdp

def choose_action(state, mdp_data, gamma):
    q_values = mdp_data['probs'][state] @ (mdp_data['rewards'] + gamma * mdp_data['value'])
    if np.isclose(q_values[0], q_values[1]):
        return np.random.choice([0, 1])
    return int(np.argmax(q_values))

def update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, reward):
    mdp_data['count_SAS'][state, action, new_state] += 1
    mdp_data['count_R'][new_state, int(reward * -1)] += 1

def update_mdp_transition_probs_reward(mdp_data):
    count_s_0 = np.sum(mdp_data['count_SAS'][:, 0, :], axis=1)
    count_s_1 = np.sum(mdp_data['count_SAS'][:, 1, :], axis=1)
    count_S_R = np.sum(mdp_data['count_R'], axis=1)
    num_states = mdp_data['num_states']
    for s in range(num_states):
        if count_s_0[s] != 0:
            mdp_data['probs'][s, 0, :] = mdp_data['count_SAS'][s, 0, :] / count_s_0[s]
        if count_s_1[s] != 0:
            mdp_data['probs'][s, 1, :] = mdp_data['count_SAS'][s, 1, :] / count_s_1[s]
        if count_S_R[s] != 0:
            mdp_data['rewards'][s] = -mdp_data['count_R'][s, 1] / count_S_R[s]

def update_mdp_value(mdp_data, tolerance, gamma):
    transition_probs = mdp_data['probs']
    value = mdp_data['value']
    reward = mdp_data['rewards']
    it = 0
    while True:
        expected = transition_probs @ (reward + gamma * value)
        new_value = np.max(expected, axis=1)
        it += 1
        if np.all(np.abs(new_value - value) < tolerance):
            mdp_data['value'] = new_value
            break
        value = new_value
    return it == 1

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train_policy():
    NUM_STATES = 163
    GAMMA = 0.995
    TOLERANCE = 0.01
    NO_LEARNING_THRESHOLD = 20
    max_failures = 500

    time = 0
    time_steps_to_failure = []
    num_failures = 0
    time_at_start_of_current_trial = 0

    cart_pole = CartPole(Physics())
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)
    state = cart_pole.get_state(state_tuple)
    mdp_data = initialize_mdp_data(NUM_STATES)

    consecutive_no_learning_trials = 0
    while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:
        action = choose_action(state, mdp_data, GAMMA)
        state_tuple = cart_pole.simulate(action, state_tuple)
        time = time + 1
        new_state = cart_pole.get_state(state_tuple)

        if new_state == NUM_STATES - 1:
            R = -1
        else:
            R = 0

        update_mdp_transition_counts_reward_counts(mdp_data, state, action, new_state, R)

        if new_state == NUM_STATES - 1:
            update_mdp_transition_probs_reward(mdp_data)
            converged_in_one_iteration = update_mdp_value(mdp_data, TOLERANCE, GAMMA)
            if converged_in_one_iteration:
                consecutive_no_learning_trials += 1
            else:
                consecutive_no_learning_trials = 0

        if new_state == NUM_STATES - 1:
            num_failures += 1
            if num_failures >= max_failures:
                break
            print('[INFO] Failure number {}'.format(num_failures))
            time_steps_to_failure.append(time - time_at_start_of_current_trial)
            time_at_start_of_current_trial = time

            x = -1.1 + np.random.uniform() * 2.2
            x_dot, theta, theta_dot = 0.0, 0.0, 0.0
            state_tuple = (x, x_dot, theta, theta_dot)
            state = cart_pole.get_state(state_tuple)
        else:
            state = new_state

    return mdp_data, time_steps_to_failure, cart_pole

# ============================================================
# ANIMATION: RENDER LEARNED POLICY
# ============================================================
def animate_policy(mdp_data, cart_pole, gamma, output_path='cartpole_animation.mp4', max_steps=500):
    """
    Run the learned policy and save an MP4 animation.
    """
    # Collect trajectory using the learned policy
    x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
    state_tuple = (x, x_dot, theta, theta_dot)
    state = cart_pole.get_state(state_tuple)

    trajectory = [state_tuple]
    for step in range(max_steps):
        action = choose_action(state, mdp_data, gamma)
        state_tuple = cart_pole.simulate(action, state_tuple)
        state = cart_pole.get_state(state_tuple)
        trajectory.append(state_tuple)
        if state == 162:  # failure state
            print(f"Policy failed at step {step}")
            break

    print(f"Collected {len(trajectory)} frames for animation")

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title('Cart-Pole with Learned Policy')
    ax.grid(True, alpha=0.3)

    # Ground line
    ax.axhline(y=0, color='brown', linewidth=2)

    # Elements to animate
    pole_line, = ax.plot([], [], 'k-', linewidth=3)
    cart_rect = patches.Rectangle((0, 0), 0.8, 0.25, linewidth=1, edgecolor='k', facecolor='cyan')
    base_rect = patches.Rectangle((0, 0), 0.02, 0.25, linewidth=1, edgecolor='k', facecolor='red')
    ax.add_patch(cart_rect)
    ax.add_patch(base_rect)

    # Action arrow
    arrow = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
    action_text = ax.text(2.2, 3.0, '', fontsize=12, fontweight='bold', color='green')
    step_text = ax.text(-2.8, 3.0, '', fontsize=12)

    def init():
        pole_line.set_data([], [])
        cart_rect.set_xy((-0.4, -0.25))
        base_rect.set_xy((-0.01, -0.5))
        arrow.set_visible(False)
        action_text.set_text('')
        step_text.set_text('')
        return pole_line, cart_rect, base_rect, arrow, action_text, step_text

    def update(frame_idx):
        state_tuple = trajectory[frame_idx]
        x, x_dot, theta, theta_dot = state_tuple

        # Pole endpoints
        pole_x = [x, x + 4 * cart_pole.length * sin(theta)]
        pole_y = [0, 4 * cart_pole.length * cos(theta)]
        pole_line.set_data(pole_x, pole_y)

        # Cart
        cart_rect.set_xy((x - 0.4, -0.25))
        base_rect.set_xy((x - 0.01, -0.5))

        # Determine action for display (recompute for this state)
        state = cart_pole.get_state(state_tuple)
        action = choose_action(state, mdp_data, gamma)

        # Action arrow
        arrow.set_visible(True)
        if action == 1:
            arrow.xy = (x + 0.6, 0.5)
            arrow.set_position((x - 0.6, 0.5))
            action_text.set_text('Push RIGHT')
            action_text.set_color('green')
        else:
            arrow.xy = (x - 0.6, 0.5)
            arrow.set_position((x + 0.6, 0.5))
            action_text.set_text('Push LEFT')
            action_text.set_color('blue')

        step_text.set_text(f'Step: {frame_idx}')

        return pole_line, cart_rect, base_rect, arrow, action_text, step_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(trajectory), interval=50, blit=False)

    anim.save(output_path, writer='ffmpeg', fps=20, dpi=150)
    plt.close(fig)
    print(f"Animation saved to: {output_path}")
    return trajectory

# ============================================================
# PLOT LEARNING CURVE
# ============================================================
def plot_learning_curve(time_steps_to_failure, output_path='learning_curve.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear
    ax = axes[0]
    ax.plot(np.arange(len(time_steps_to_failure)), time_steps_to_failure, 'k-', alpha=0.5)
    window = 30
    if len(time_steps_to_failure) >= window:
        w = np.array([1.0/window for _ in range(window)])
        weights = lfilter(w, 1, np.array(time_steps_to_failure))
        x = np.arange(window//2, len(time_steps_to_failure) - window//2)
        ax.plot(x, weights[window:len(time_steps_to_failure)], 'r--', linewidth=2)
    ax.set_xlabel('Failure number')
    ax.set_ylabel('Steps to failure')
    ax.set_title('Learning Curve')
    ax.grid(True, alpha=0.3)

    # Log
    ax = axes[1]
    log_tstf = np.log(np.array(time_steps_to_failure))
    ax.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k-', alpha=0.5)
    if len(time_steps_to_failure) >= window:
        w = np.array([1.0/window for _ in range(window)])
        weights = lfilter(w, 1, log_tstf)
        x = np.arange(window//2, len(log_tstf) - window//2)
        ax.plot(x, weights[window:len(log_tstf)], 'r--', linewidth=2)
    ax.set_xlabel('Failure number')
    ax.set_ylabel('Log of steps to failure')
    ax.set_title('Learning Curve (Log Scale)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to: {output_path}")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TRAINING POLICY...")
    print("=" * 60)
    mdp_data, time_steps_to_failure, cart_pole = train_policy()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total failures: {len(time_steps_to_failure)}")
    print(f"Mean steps (first 50): {np.mean(time_steps_to_failure[:50]):.1f}")
    print(f"Mean steps (last 50):  {np.mean(time_steps_to_failure[-50:]):.1f}")
    print(f"Max steps: {np.max(time_steps_to_failure)}")

    print("\n" + "=" * 60)
    print("GENERATING PLOTS & ANIMATION...")
    print("=" * 60)
    
    # Use relative paths so it works on any OS
    import os
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_learning_curve(time_steps_to_failure, os.path.join(output_dir, 'learning_curve.png'))
    animate_policy(mdp_data, cart_pole, 0.995,
                   output_path=os.path.join(output_dir, 'cartpole_animation.mp4'),
                   max_steps=500)
    print("\nDone!")