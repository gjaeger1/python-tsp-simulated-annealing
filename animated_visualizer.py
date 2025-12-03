import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animateTSP(history, points, temp_history=None, weight_history=None):
    """animate the solution over time with temperature and cost function graphs

    Parameters
    ----------
    history : list
        history of the solutions chosen by the algorithm
    points: array_like
        points with the coordinates
    temp_history: list, optional
        history of temperatures during the annealing process
    weight_history: list, optional
        history of cost function values during the annealing process
    """

    # Print len() of histories for debugging
    print(f"Length of history: {len(history)}")
    print(f"Length of temp_history: {len(temp_history) if temp_history else 'N/A'}")
    print(
        f"Length of weight_history: {len(weight_history) if weight_history else 'N/A'}"
    )

    # approx 1500 frames for animation
    key_frames_mult = max(1, len(history) // 9000)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TSP Simulated Annealing Visualization", fontsize=16)

    # Subplot 1: TSP Path
    ax1.set_title("TSP Path")
    (line,) = ax1.plot([], [], "b-", lw=2, label="Current Path")

    # Subplot 2: Temperature
    ax2.set_title("Temperature")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Temperature")
    (temp_line,) = ax2.plot([], [], "r-", lw=2)
    (temp_points,) = ax2.plot([], [], "ro", markersize=4)

    # Subplot 3: Cost Function
    ax3.set_title("Cost Function")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Cost")
    (cost_line,) = ax3.plot([], [], "g-", lw=2)
    (cost_points,) = ax3.plot([], [], "go", markersize=4)

    def init():
        """initialize all plots"""
        # Initialize TSP path
        x = [points[i][0] for i in history[0]]
        y = [points[i][1] for i in history[0]]
        ax1.plot(x, y, "co", markersize=8, label="Cities")

        # Draw axes slightly bigger
        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax1.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax1.set_ylim(min(y) - extra_y, max(y) + extra_y)
        ax1.legend()

        # Initialize path to be empty
        line.set_data([], [])

        # Initialize temperature plot
        if temp_history:
            ax2.set_xlim(0, len(temp_history))
            ax2.set_ylim(min(temp_history) * 0.9, max(temp_history) * 1.1)
            ax2.grid(True, alpha=0.3)
        temp_line.set_data([], [])
        temp_points.set_data([], [])

        # Initialize cost function plot
        if weight_history:
            ax3.set_xlim(0, len(weight_history))
            ax3.set_ylim(min(weight_history) * 0.95, max(weight_history) * 1.05)
            ax3.grid(True, alpha=0.3)
        cost_line.set_data([], [])
        cost_points.set_data([], [])

        return line, temp_line, temp_points, cost_line, cost_points

    def update(frame):
        """update all plots for each frame"""
        actual_frame = frame * key_frames_mult

        # Update TSP path
        if actual_frame < len(history):
            x = [
                points[i, 0] for i in history[actual_frame] + [history[actual_frame][0]]
            ]
            y = [
                points[i, 1] for i in history[actual_frame] + [history[actual_frame][0]]
            ]
            line.set_data(x, y)

        # Update temperature plot
        if temp_history and actual_frame < len(temp_history):
            iterations = list(range(actual_frame + 1))
            temps = temp_history[: actual_frame + 1]
            temp_line.set_data(iterations, temps)
            # Show current point
            temp_points.set_data([actual_frame], [temp_history[actual_frame]])

            # Add current temperature as text
            ax2.clear()
            ax2.plot(iterations, temps, "r-", lw=2)
            ax2.plot(actual_frame, temp_history[actual_frame], "ro", markersize=6)
            ax2.set_title(f"Temperature: {temp_history[actual_frame]:.6f}")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Temperature")
            ax2.grid(True, alpha=0.3)
            if len(temp_history) > 0:
                ax2.set_xlim(0, len(temp_history))
                ax2.set_ylim(min(temp_history) * 0.9, max(temp_history) * 1.1)

        # Update cost function plot
        if weight_history and actual_frame < len(weight_history):
            iterations = list(range(actual_frame + 1))
            weights = weight_history[: actual_frame + 1]
            cost_line.set_data(iterations, weights)
            # Show current point
            cost_points.set_data([actual_frame], [weight_history[actual_frame]])

            # Add current cost as text
            ax3.clear()
            ax3.plot(iterations, weights, "g-", lw=2)
            ax3.plot(actual_frame, weight_history[actual_frame], "go", markersize=6)
            ax3.set_title(f"Cost: {weight_history[actual_frame]:.2f}")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Cost")
            ax3.grid(True, alpha=0.3)
            if len(weight_history) > 0:
                ax3.set_xlim(0, len(weight_history))
                ax3.set_ylim(min(weight_history) * 0.95, max(weight_history) * 1.05)

        return line, temp_line, temp_points, cost_line, cost_points

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(history), key_frames_mult),
        init_func=init,
        interval=50,  # Slightly slower for better visibility
        repeat=False,
        blit=False,  # Disable blitting since we're clearing axes
    )

    plt.tight_layout()
    plt.show()
