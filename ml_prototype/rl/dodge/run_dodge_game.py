from ml_prototype.rl.dodge.dodge_game import DodgeGame
from ml_prototype.rl.dodge.qlearn import update_q, select_action
import pygame


def play_and_learn_dodge_game(
    episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.3, epsilon_decay=0.995, 
    visualize=False, delay=100
):
    env = DodgeGame()
    total_rewards = []
    best_reward = float("-inf")
    episode_stats = {"collisions": 0, "avg_steps": 0, "good_episodes": 0}

    try:
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_actions = []

            while not done and steps < 100:
                if visualize:
                    env.render()
                    pygame.time.delay(delay)  # Add delay for visualization
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            return

                action = select_action(state)
                next_state, reward, done = env.step(action)
                episode_actions.append((action, reward))

                # Update Q-values
                update_q(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

                # Early termination for very poor performance
                if steps > 10 and total_reward < -50:
                    break

            # Update statistics
            if done and total_reward <= -20:  # Collision occurred
                episode_stats["collisions"] += 1
            if total_reward > 0:  # Good episode
                episode_stats["good_episodes"] += 1
            episode_stats["avg_steps"] = (
                episode_stats["avg_steps"] * episode + steps
            ) / (episode + 1)

            # Detailed episode logging
            if episode % 100 == 0:
                print(f"\nEpisode {episode} details:")
                print(f"Steps: {steps}, Total reward: {total_reward}")
                print(f"Action history: {episode_actions}")
                print(f"Stats: Collisions: {episode_stats['collisions']}")
                print(f"Good episodes: {episode_stats['good_episodes']}")
                print(f"Average steps: {episode_stats['avg_steps']:.2f}")

            # Decay epsilon
            epsilon = max(0.01, epsilon * epsilon_decay)

            total_rewards.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                print(f"New best reward: {best_reward}")

            if episode % 100 == 0:
                avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards))
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

    finally:
        if visualize:
            env.close()

    # Final statistics
    print("\nTraining Summary:")
    print(f"Total collisions: {episode_stats['collisions']}")
    print(f"Good episodes: {episode_stats['good_episodes']}")
    print(f"Average steps per episode: {episode_stats['avg_steps']:.2f}")
    print(f"Best reward achieved: {best_reward}")
    print(f"Training completed. Final average reward: {sum(total_rewards[-100:]) / 100:.2f}")


if __name__ == "__main__":
    # Train first
    play_and_learn_dodge_game(episodes=1000)
    # Then visualize the trained agent
    play_and_learn_dodge_game(episodes=5, epsilon=0.01, visualize=True, delay=300)
