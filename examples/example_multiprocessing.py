"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.legged_robot.B1 import B1
from pyrep.objects.shape import Shape
import numpy as np
from multiprocessing import Process

SCENE_FILE = join(dirname(abspath(__file__)),
                  'B1_4_4.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 1
EPISODE_LENGTH = 10

# Multi Process
PROCESSES = 2

class B1Env(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        self.agent = B1()
        # self.agent.set_control_loop_enabled(False)
        # self.agent.set_motor_locked_at_zero_velocity(True)
        # self.body = self.agent.get_handle('B1')
        # self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions(),
                               self.agent.get_joint_velocities()])
                              #  self.target.get_position()])

    def reset(self):
        # Get a random position within a cuboid and set the target position
        # pos = list(np.random.uniform(POS_MIN, POS_MAX))
        # self.body.set_position([0,0,1])
        self.agent.set_joint_positions(self.initial_joint_positions)
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_positions(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        # ax, ay, az = self.agent_ee_tip.get_position()
        # tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        # reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
        reward = 0
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(12,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass

def run():
  # pr = PyRep()
  # pr.launch(SCENE_FILE, headless=True)
  # pr.start()
  # Do stuff... ##########
  env = B1Env()
  agent = Agent()
  replay_buffer = []

  for e in range(EPISODES):

      print('Starting episode %d' % e)
      state = env.reset()
      for i in range(EPISODE_LENGTH):
          print(i)
          action = agent.act(state)
          reward, next_state = env.step(action)
          # replay_buffer.append((state, action, reward, next_state))
          state = next_state
          # agent.learn(replay_buffer)

  print('Done!')

  ########################
  env.shutdown()
  

processes = [Process(target=run, args=()) for i in range(PROCESSES)]
[p.start() for p in processes]
[p.join() for p in processes]


#     # CPG
#     MI = 0.05
#     self.w11, self.w22 = 1.4, 1.4
#     self.w12 =  0.18 + MI
#     self.w21 = -0.18 - MI
    
#     self.o1 = 0.01
#     self.o2 = 0.01
    
#     # Graph
#     self.cpg1 = sim.addGraphStream(self.graph, 'o1', '')
#     self.cpg2 = sim.addGraphStream(self.graph, 'o2', '', 0, [0, 1, 0])
    
#     # robot constant
#     # Walking
#     #self.h_b = 0.3
#     #self.t_b = 0.5
#     #self.k_b = -1
    
#     #self.h_g = 1.0
#     #self.t_g = 0.3
#     #self.k_g = 0.1
    
#     # bias
#     # hip joint
#     self.h_b = 0.3
#     # thigh joint
#     self.Ft_b = 0.5
#     self.Rt_b = 0.5
#     # knee joint
#     self.Fk_b = -1
#     self.Rk_b = -1
    
#     # gain
#     # hip joint
#     self.h_g = 1.0
#     # thigh joint
#     self.Ft_g = 0.3
#     self.Rt_g = 0.3
#     # knee joint
#     self.Fk_g = 0.1
#     self.Rk_g = 0.1  
  
  
#     cpg_delay = 100
#     self.d1 = deque([0]*cpg_delay)
#     self.d2 = deque([0]*cpg_delay)
#     print(self.d1)
    
#     for i in range(100):
#         self.o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2)
#         self.o2 = math.tanh(self.w22*self.o2 + self.w21*self.o1)
#         self.d1.append(self.o1)
#         o1_d = self.d1.popleft()

#         self.d2.append(self.o2)
#         o2_d = self.d2.popleft()
    

# def sysCall_actuation():
#     # put your actuation code here
#     self.o1 = math.tanh(self.w11*self.o1 + self.w12*self.o2)
#     self.o2 = math.tanh(self.w22*self.o2 + self.w21*self.o1)
    
#     o1 = self.o1
#     o2 = self.o2
    
#     self.d1.append(o1)
#     o1_d = self.d1.popleft()

#     self.d2.append(o2)
#     o2_d = self.d2.popleft()
#     #print(self.d)
#     #print(d1)
                
#     #Joint Command
#     jointPosTarget = [0]*self.jointNum
    
#     # command to hip joint
#     jointPosTarget = [0]*self.jointNum
#     for i in range(0, 12, 3):
#         #jointPosTarget[i] = self.h_b
#         if i == 0 : # FL thigh
#             jointPosTarget[i] = o1*self.Ft_g+self.h_b
#         if i == 9: # RL thigh
#             jointPosTarget[i] = o1*self.Rt_g+self.h_b
#         if i == 3: # RR thigh
#             jointPosTarget[i] = -o1_d*self.Rt_g+self.h_b
#         if i == 6: # FR thigh
#             jointPosTarget[i] = -o1_d*self.Ft_g+self.h_b
        
#     # command to thight joint
#     for i in range(1, 12, 3):
#         if i == 1 : # FL thigh
#             jointPosTarget[i] = o1*self.Ft_g+self.Ft_b
#         if i == 10: # RL thigh
#             jointPosTarget[i] = o1*self.Rt_g+self.Rt_b
#         if i == 4: # RR thigh
#             jointPosTarget[i] = o1_d*self.Rt_g+self.Rt_b
#         if i == 7: # FR thigh
#             jointPosTarget[i] = o1_d*self.Ft_g+self.Ft_b

#     # command to knee joint
#     for i in range(2, 12, 3):
#         if i == 2: # FL knee 
#             jointPosTarget[i] = o2*self.Fk_g+self.Fk_b
#         if i == 11: # RL knee 
#             jointPosTarget[i] = o2*(self.Rk_g)+self.Rk_b
#         if i == 5 : # RR knee 
#             jointPosTarget[i] = o2_d*(self.Rk_g)+self.Rk_b
#         if i == 8: # FR knee 
#             jointPosTarget[i] = o2_d*(self.Fk_g)+self.Fk_b