import re
import matplotlib.pyplot as plt
import numpy as np

# 打开文本文件并读取所有内容
with open('/home/clothsim/softgym/data/train1/reward.txt', 'r') as f:
    data = f.read()

episodes = []
episode_reward = []
for match in re.finditer(r'Episode: (\d+)/\d+  \| Episode Reward: ([\d.-]+)', data):
    episode_num, reward = match.groups()
    if not match.string.startswith('Test'):
        episodes.append(int(episode_num))
        episode_reward.append(float(reward))


plt.plot(episodes, episode_reward, '-o')
plt.title('Episode Reward vs Episode')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.ylim(-2,-0.5)
plt.show()

del episodes[:], episode_reward[:]

episodes = []
episode_reward = []
for match in re.finditer(r'Test Episode: (\d+)/\d+  \| Episode Reward: ([\d.-]+)', data):
    episode_num, reward = match.groups()
    episodes.append(int(episode_num))
    episode_reward.append(float(reward))


plt.plot(episodes, episode_reward, '-o')
plt.title('Tetst Episode Reward vs Episode')
plt.xlabel('Test Episode')
plt.ylabel('Episode Reward')
plt.ylim(-2,-0.5)
plt.show()

del episodes[:], episode_reward[:]

episodes = []
wrinkle_density = []
wrinkle_average_depth = []
lines = data.split("\n")
for i in range(len(lines) - 1):
    match = re.search(r'Episode: (\d+)/\d', lines[i])
    if match:
        episode_num = match.groups()
        if not lines[i].startswith('Test'):
            next_line = lines[i+1]
            match2 = re.search(r'wrinkle desity: ([\d.]+)  \| wrinkle averange depth: ([\d.-]+)', next_line)
            if match2:
                density, depth = match2.groups()
                episodes.append(int(episode_num[0]))
                wrinkle_density.append(float(density))
                wrinkle_average_depth.append(float(depth))

plt.plot(episodes, wrinkle_density, '-s', label='Wrinkle Density')
plt.plot(episodes, wrinkle_average_depth, '-^', label='Wrinkle Average Depth')
plt.title('Wrinkle Information vs Episode')
plt.xlabel('Episode')
plt.ylabel('Wrinkle Density/Average Depth')
plt.legend()
plt.show()