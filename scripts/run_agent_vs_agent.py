from typing import Tuple, Dict, List, Any

from kaggle_environments import make

if __name__ == "__main__":

    env = make("football", debug=True,
               configuration={"save_video": True,
                              "scenario_name": "11_vs_11_kaggle"})

    # Define players
    left_player = "main.py"  # A custom agent, eg. random_agent.py or example_agent.py
    right_player = "main.py"  # eg. A built in 'AI' agent
    # right_player = "main.py"

    # Run the whole sim
    # Output returned is a list of length n_steps. Each step is a list containing the output for each player as a dict.
    # steps
    output: List[Tuple[Dict[str, Any], Dict[str, Any]]] = env.run([left_player, right_player])

    for s, (left, right) in enumerate(output):
        print(f"\nStep {s}")

        print(f"Left player ({left_player}): \n"
              f"actions taken: {left['action']}, "
              f"reward: {left['reward']}, "
              f"status: {left['status']}, "
              f"info: {left['info']}")

        print(f"Right player ({right_player}): \n"
              f"actions taken: {right['action']}, "
              f"reward: {right['reward']}, "
              f"status: {right['status']}, "
              f"info: {right['info']}\n")

    print(f"Final score: {sum([r['reward'] for r in output[0]])} : {sum([r['reward'] for r in output[1]])}")

    env.render(mode="human", width=800, height=600)
