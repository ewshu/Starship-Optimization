import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class StarshipOptimizer:
    def __init__(self):
        # Physical parameters
        self.g = 9.81
        self.m = 100000  # kg (total mass)
        self.length = 50  # m
        self.I = (1 / 12) * self.m * self.length ** 2  # moment of inertia

        # Engine parameters
        self.max_thrust = 2210000  # N (single Raptor)
        self.min_throttle = 0.4
        self.max_gimbal = np.deg2rad(20)

        # Time parameters
        self.N = 400  # number of timesteps
        self.dt = 0.04  # timestep [s]

    def setup_optimization(self):
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(6, self.N)
        U = opti.variable(2, self.N - 1)

        # Initial and final conditions
        opti.subject_to(X[:, 0] == ca.vertcat(0, 0, 1000, -80, np.pi / 2, 0))
        opti.subject_to(X[:, -1] == ca.vertcat(0, 0, 0, 0, 0, 0))

        # Dynamics constraints
        for i in range(self.N - 1):
            x_next = self.dynamics_step(X[:, i], U[:, i])
            opti.subject_to(X[:, i + 1] == X[:, i] + x_next * self.dt)

            # Control bounds
            opti.subject_to(opti.bounded(self.min_throttle, U[0, i], 1))
            opti.subject_to(opti.bounded(-self.max_gimbal, U[1, i], self.max_gimbal))

        # Objective
        cost = 0
        for i in range(self.N - 1):
            cost += U[0, i] ** 2  # Minimize thrust
            cost += 100 * U[1, i] ** 2  # Minimize gimbal angle
            cost += 200 * X[5, i] ** 2  # Minimize angular velocity

        opti.minimize(cost)

        # Solver options
        opts = {
            'ipopt.max_iter': 5000,
            'ipopt.tol': 1e-4,
            'print_time': 0
        }
        opti.solver('ipopt', opts)

        return opti, X, U

    def dynamics_step(self, state, control):
        x_dot = state[1]
        y_dot = state[3]
        theta = state[4]
        theta_dot = state[5]

        thrust = control[0] * self.max_thrust
        gamma = control[1]

        F_x = thrust * ca.sin(theta + gamma)
        F_y = thrust * ca.cos(theta + gamma)

        x_ddot = F_x / self.m
        y_ddot = F_y / self.m - self.g

        torque = -self.length / 2 * thrust * ca.sin(gamma)
        theta_ddot = torque / self.I

        return ca.vertcat(x_dot, x_ddot, y_dot, y_ddot, theta_dot, theta_ddot)

    def solve(self):
        print("Setting up optimization problem...")
        opti, X, U = self.setup_optimization()

        print("Solving...")
        try:
            sol = opti.solve()
            print("Solution found!")
            return np.array(sol.value(X)).T, np.array(sol.value(U)).T
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None, None

    def animate_solution(self, X, U):
        # Configure matplotlib for macOS
        import matplotlib
        matplotlib.use('TkAgg')

        # Set up the figure with dark theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)

        # Configure axes and style
        ax.set_xlim(-200, 200)
        ax.set_ylim(-100, 1100)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Create lines
        rocket, = ax.plot([], [], color='cyan', lw=2)
        trail, = ax.plot([], [], 'white', ls='--', alpha=0.3, lw=1)

        def update(frame):
            # Get current state
            x = X[frame, 0]
            y = X[frame, 2]
            theta = X[frame, 4]

            # Rocket points
            x_points = [x - (self.length / 2) * np.sin(theta),
                        x + (self.length / 2) * np.sin(theta)]
            y_points = [y - (self.length / 2) * np.cos(theta),
                        y + (self.length / 2) * np.cos(theta)]

            # Update rocket position
            rocket.set_data(x_points, y_points)

            # Update trail (only show last 50 points for a fading effect)
            start_idx = max(0, frame - 50)
            trail.set_data(X[start_idx:frame + 1, 0], X[start_idx:frame + 1, 2])

            # Add velocity info
            if frame < len(U):
                velocity = np.sqrt(X[frame, 1] ** 2 + X[frame, 3] ** 2)
                ax.set_title(f'Time: {frame * self.dt:.1f}s | Velocity: {velocity:.1f} m/s',
                             color='white', pad=20)

            return rocket, trail

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(X),
                                       interval=50,
                                       blit=True)

        # Save animation using matplotlib's animation writer
        try:
            writer = animation.FFMpegWriter(fps=20, bitrate=2000)
            anim.save('starship_landing.mp4', writer=writer)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Failed to save animation: {e}")
            print("But the animation will still be displayed...")

        plt.show()
        return anim