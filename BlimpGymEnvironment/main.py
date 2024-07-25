from blimp import Blimp

"""
BlimpEnv takes the path to the model as the input, but by default it uses the Sano Blimp which is used
for this project.
render_mode paramater is used to specify how to render what is happening,
* Human = Dispaly everything
* Graph = Show graph of how the reward is changing over time
"""
env = Blimp(render_mode="human")

"""
Env reset will reset the model to it's original state, and returns the starting state of the model/environment
It will have other misc information which won't be going to the network.
"""
state, info = env.reset()

"""
Evn step function will take in action and evaluate the action, and return it's next observation and
the reward the action generated. Similar to the reset it will provide other information that can use useful
for debugging and other purposes.
"""
observation, reward, terminated, info = env.step([0,0,0,0])

for i in range(500):
    # print([round(x,2) for x in observation[3]])
    # env.render()
    observation, reward, terminated, info = env.step([1,-1,1,1])

print([round(x,2) for x in observation[3]])
