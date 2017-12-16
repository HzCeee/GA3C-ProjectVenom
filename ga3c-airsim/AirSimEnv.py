# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc
from gym.utils import seeding
from Config import Config
from GameManager import GameManager
import math, io, random
from PIL import Image
import threading
#from PythonClient import *
from projection import *
from gym import spaces
from collections import deque

locker = threading.Lock()
global_port = 41451

g = 9.8



class Environment:
    def __init__(self, discrete=True):
        global locker
        global global_port
        locker.acquire(True)
        this_port = global_port
        global_port += 1
        locker.release()    

        #self.client = AirSimClient(port=this_port)
        #self.client.confirmConnection()
        #self.client.enableApiControl(True)
        #self.client.armDisarm(True)

        self.log_file = open('logs.txt', 'w')
        self.acc_file = open('accs.txt', 'w')

        self.episodes = 0
        self.fps = 100
        self.max_iter = 15*self.fps

        self.c = np.matrix([0.0, 0.0, 10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.thrust = 9.8 
        self.mass = 1
        self.v = np.matrix([0.0, 0.0, 0.0])

        self.t = np.matrix([10.0, 0.0, 10.0])
        self.o_t = np.matrix([0.0, 0.0, 0.0]) 
        self.thrust_t = 9.8
        self.mass_t = 1
        self.v_t = np.matrix([0.0, 0.0, 0.0])

        self.queue_len = 8
        self.coord_queue = None
        self.dist_queue = None
        self.width = 256
        self.height = 144
        self.image = None
        self.iteration = 0

        self.current_state = self._get_obs()
        self.reset()

        self.viewer = None
        self.discrete = discrete
        if self.discrete:
            self.action_space = spaces.Discrete(81)
            self.observation_space = spaces.Box(low=np.zeros(self.current_state.shape),
                                                high=np.zeros(self.current_state.shape) + 1)
        else: 
            self.action_space = spaces.Box(-1, 1, shape = (4,))
            self.observation_space = spaces.Box(low=np.zeros(self.current_state.shape),
                                            high=np.zeros(self.current_state.shape) + 1)
        self.current_state = None

        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        self.game = GameManager('AirSim', display=Config.PLAY_MODE, custom_env=self)
        #self._seed()

    def get_num_actions(self):
        return self.game.env.action_space.n


    ##
    ## Helper Functions ##
    ##

    # f=ma
    # a = f/m
    # v = v+a
    # c = c+v
    def move(self, o, thrust, mass):
        A = radians(o.item(2))
        B = radians(o.item(1))
        C = radians(o.item(0))
        f = np.matrix([0,0,float(thrust)/float(mass), 1])
        R = np.matrix([[cos(A)*cos(B), cos(A)*sin(B)*sin(C)-sin(A)*cos(C), cos(A)*sin(B)*cos(C) + sin(A)*sin(C), 0],
                       [sin(A)*cos(B), sin(A)*sin(B)*sin(C)+cos(A)*cos(C), sin(A)*sin(B)*cos(C) - cos(A)*sin(C), 0],
                       [-sin(B),       cos(B)*sin(C),                      cos(B)*cos(C),                        0],
                       [0,             0,                                  0,                                    1]])
        a = R*np.transpose(f)
        a = np.transpose(a[:3])
        return a
        
    # If dimensions change they need to be updated in:
    # ThreadPredictor.py:47
    # NetworkVP.py:134 
    def _get_obs(self):
        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        pix = np.array((x/256.0,y/144.0)).flatten()
        d = np.array([np.linalg.norm(self.t-self.c)/30.0])
        if self.coord_queue is None:
            self.coord_queue = deque([pix]*self.queue_len)
            self.dist_queue = deque([d]*self.queue_len)
        else:
            self.coord_queue.append(pix)
            self.coord_queue.popleft() 
            self.dist_queue.append(d)
            self.dist_queue.popleft()
        coords = np.concatenate(list(self.coord_queue))
        dists = np.concatenate(list(self.dist_queue))
        self.current_state = np.concatenate([np.array(self.o).flatten()/(360.0), # 3
                                             np.array(self.thrust).flatten()/float(2*self.mass*g),
                                             #np.array(self.t).flatten()/30.0, # 3
                                             #np.array(self.c).flatten()/30.0, # 3
                                             #d/30.0, # 1
                                             coords, # 8
                                             dists
                                            ], 0)
        #self.current_state = np.array(d)
        return self.current_state

    ##
    ## End Helper Functions ##
    ##

    def reset(self):
        self.iteration = 0
        self.c = np.matrix([0.0, 0.0, 10.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.thrust = 9.8
        self.mass = 1
        self.v = np.matrix([0.0, 0.0, 0.0])

        self.t = np.matrix([10.0, 0.0, 10.0])+np.random.normal(0, 5)
        self.o_t = np.matrix([0.0, 0.0, 0.0])
        self.thrust_t = 9.8
        self.mass_t = 1
        self.v_t = np.matrix([0.0, 0.0, 0.0])

        
        (x, y), _ = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height)) 
        d = np.linalg.norm(self.c-self.t)
        while (x > self.width or x < 0 or y > self.height or y < 0 or d > 30 or d < 5):
            self.t = np.matrix([10.0, 0.0, 10.0])+np.random.normal(0, 5)
            (x,y), _ = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
            d = np.linalg.norm(self.c-self.t)
        self.v = np.matrix([0.0, 0.0, 0.0])
        self.v_t = np.matrix([0.0, 0.0, 0.0])
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.nb_correct = 0
        self.image = None
        #self.fw = None
        self.last_d = d
        self.start_d = np.linalg.norm(self.t-self.c)
        self.current_state = self._get_obs()
        return self.current_state

    '''
action	 roll	 pitch	 yaw	thrust
0	-1	-1	-1	
1	-1	-1	 0
2	-1	-1	 1
3	-1	 0	-1
4	-1	 0	 0
5	-1	 0	 1
6	-1	 1	-1
7	-1	 1	 0
8	-1	 0	 1
9	 0	-1	-1
10	 0	-1	 0
11	 0	-1	 1
12	 0	 0	-1
13	 0	 0	 0
14	 0	 0	 1
15	 0	 1	-1
16	 0 	 1	 0
17	 0	 0	 1
18	 1	-1	-1
19	 1	-1	 0
20	 1	-1	 1
21	 1	 0	-1
22	 1	 0	 0
23	 1	 0	 1
24	 1	 1	-1
25	 1	 1	 0
26	 1	 0	 1
    '''
    def step(self, raw_action):
        j = 0
        dyaw = raw_action % 3 - 1
        raw_action = int(raw_action)/int(3)
        dpitch = raw_action % 3 - 1
        raw_action = int(raw_action)/int(3)
        droll = raw_action % 3 - 1
        raw_action = int(raw_action)/int(3)
        dthrust = raw_action % 3 - 1 

        self.o += np.matrix([60.0*droll/self.fps, 60.0*dpitch/self.fps, 60.0*dyaw/self.fps])
        self.thrust += 10*dthrust/self.fps
        if self.thrust > 2*g*self.mass: self.thrust = 2*g*self.mass
        acc = self.move(self.o, self.thrust, self.mass) + np.matrix([0.0,0.0,-self.mass*g])
#        acc = np.matrix([x,y,z])

        max_v = 10.0/self.fps

        self.v = self.v + acc/self.fps
        if np.linalg.norm(self.v) > max_v: self.v *= max_v/np.linalg.norm(self.v)
        self.c = self.c + self.v


        self.o_t += np.random.normal(0, 30.0/self.fps)
        self.thrust_t += np.random.normal(0,5.0/self.fps) 
        t_acc = self.move(self.o_t, self.thrust_t, self.mass_t)
        #t_acc = np.random.normal(0, 1)
        if np.linalg.norm(t_acc) > 1: t_acc /= np.linalg.norm(t_acc)
        self.v_t = self.v_t + t_acc/self.fps
        if np.linalg.norm(self.v_t) > max_v: self.v *= max_v/np.linalg.norm(self.v_t)
        #self.t = self.t + self.v_t 
 
        self.previous_state = self.current_state
        self.current_state = self._get_obs()
       
        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        d = np.linalg.norm(self.t-self.c)
        
        ## r1
        self.reward = self.last_d - d
        self.last_d = d
        self.done = (self.iteration > self.max_iter)
#        if x > self.width or x < 0 or y > self.height or y < 0 or d > 30 or self.iteration > 500:
        if d > 30 or self.iteration > self.max_iter:
            self.done = True
            self.reward = 0

        in_view = [ (pix[0] < 1 and pix[0] > 0 and pix[1] < 1 and pix[1] > 0) for pix in self.coord_queue] 
        if True not in in_view:
            self.done = True
            self.reward = 0

        if d < 1:
            self.done = True
            self.reward = 100

        self.total_reward += self.reward
        self.iteration += 1

        if self.done is True:
            self.episodes += 1
            self.total_reward = 0
        return self.reward, self.done
