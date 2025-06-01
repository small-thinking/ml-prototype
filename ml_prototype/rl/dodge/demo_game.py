from ml_prototype.rl.dodge.run_dodge_game import play_and_learn_dodge_game


def main():
    # First train the agent
    print("Training the agent...")
    play_and_learn_dodge_game(episodes=1000, epsilon=0.3)
    
    # Then demonstrate the trained agent
    print("\nStarting demonstration...")
    play_and_learn_dodge_game(
        episodes=10,
        epsilon=0.01,  # Very low exploration rate for demonstration
        visualize=True,
        delay=300  # 300ms delay between steps
    )


if __name__ == "__main__":
    main()
