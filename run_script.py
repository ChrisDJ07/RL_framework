import subprocess
import sys

def run_visualization():
    command = [
        sys.executable,
        "utils/visualize_routes.py",
        "--checkpoint-path", "checkpoints/no_hazard_control_1/best_model.pt",
        "--save-path", "results/visualization_runs/my_routes_control_1.png",
        "--config-path", "configs/no_hazard_training/no_hazard_config_control_1.json",
        "--num-episodes", "6",
        "--cols", "3"
    ]

    try:
        subprocess.run(command, check=True)
        print("Visualization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running visualization: {e}")

if __name__ == "__main__":
    run_visualization()