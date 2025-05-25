import gym
import time
# 创建环境
env = gym.make("CartPole-v1")

# 初始化环境
state = env.reset()
done = False

# 可视化环境的初始状态
env.render()

while not done:
    # 随机选择一个动作（0 或 1）
    action = env.action_space.sample()
    
    # 执行一步动作
    next_state, reward, done, info = env.step(action)
    
    # 可视化当前状态
    env.render()

    # 打印相关信息
    print(f"State: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # 更新状态
    state = next_state

    time.sleep(0.05)

# 关闭环境
env.close()