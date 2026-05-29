import metaworld
import gymnasium as gym

# Create MT1 benchmark (single task)
mt1 = metaworld.MT1('reach-v3')

# Get one environment
env = mt1.train_classes['reach-v3']()
env.render_mode = 'human'  # optional, for visualization

# Set a specific task
task = mt1.train_tasks[0]
env.set_task(task)

# Reset environment
done = False
obs, info = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

env.close()