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

class Environment:
    '''
    def __init__(self):
        self.game = GameManager(Config.ATARI_GAME, display=Config.PLAY_MODE)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()
    '''
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
        self.fps = 60
        self.max_iter = 15*self.fps

        self.t = np.matrix([10.0, 0.0, 0.0])
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([0.0, 0.0, 0.0])
        self.v = np.matrix([0.0, 0.0, 0.0])
        self.v_t = np.matrix([0.0, 0.0, 0.0])
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.queue_len = 4
        self.coord_queue = None
        self.width = 256
        self.height = 144
        self.image = None
        self.iteration = 0

        #self._render()
        self.current_state = self._get_obs()
        self.reset()

        self.viewer = None
        self.discrete = discrete
        if self.discrete:
            self.action_space = spaces.Discrete(27)
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
        self._seed()

    ##
    ## Helper Functions ##
    ##
    def _random_orientation(self, t):
        i = 51
        while i > 50:
            while True:
                c = np.matrix([random.normalvariate(t.item(0), 10),
                               random.normalvariate(t.item(1), 10),
                               random.normalvariate(t.item(2), 5)])
                d = np.linalg.norm(t - c)
                if d > 10 and d < 15 and c.item(2) < -5:
                    #    break
                    # while True:
                    # o = np.matrix([random.uniform(-180,180),
                    #               random.uniform(-180,180),
                    #               random.uniform(-180,180)])
                    o = get_o_from_pts(t, c)
                    (x, y), target_in_front = projection(t, c, o, w=float(self.width), h=float(self.height))
                    if x == self.width / 2 and y == self.height / 2 and target_in_front:
                        break

            r = np.matrix([0.0, 0.0, 0.0])
            v = np.matrix([0.0, 0.0, 0.0])

            for i in range(50):
                j = 0
                while True:
                    rot_inc = 5.0 + float(j) / 10.0
                    vel_inc = 10.0 + float(j) / 10.0
                    if j > 50:
                        break
                    dC = np.matrix([random.normalvariate(v.item(0), vel_inc / self.fps),
                                    random.normalvariate(v.item(1), vel_inc / self.fps),
                                    random.normalvariate(v.item(2), vel_inc / self.fps)]
                                   )
                    dO = np.matrix([random.normalvariate(r.item(0), vel_inc / self.fps),
                                    random.normalvariate(r.item(1), rot_inc / self.fps),
                                    random.normalvariate(r.item(2), rot_inc / self.fps)]
                                   )
                    newC = np.add(c, dC)
                    newO = np.add(o, dO)
                    d = np.linalg.norm(self.t - newC)
                    (x, y), target_in_front = projection(self.t, newC, newO, w=float(self.width),
                                                         h=float(self.height))
                    total_v = np.linalg.norm(dC)
                    if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and y <= float(
                            self.height) * 0.95 and y >= float(self.height) * 0.05 \
                            and d > 10 and d < 15 and newC.item(2) < -5 \
                            and total_v * self.fps <= 30 \
                            and target_in_front:
                        break
                    j += 1
                c = newC
                v = dC
                o = newO
                r = dO
            if x <= float(self.width) * 0.95 and x >= float(self.width) * 0.05 and y <= float(
                    self.height) * 0.95 and y >= float(self.height) * 0.05 \
                    and d > 10 and d < 15 and c.item(2) < -5 \
                    and target_in_front:
                    break
        self.last_d = d = np.linalg.norm(self.t - c)

        (x, y), target_in_front = projection(self.t, c, o, w=float(self.width), h=float(self.height))
        return (c, o)

    def _get_rgb(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        # rgb_vec = rgb.flatten()
        return rgb

    def _get_depth(self, response):
        binary_rgb = response.image_data_uint8
        png = Image.open(io.BytesIO(binary_rgb)).convert('RGB')
        rgb = np.array(png)
        depth = np.expand_dims(rgb[:, :, 0], axis=2)
        # w = Image.fromarray(depth, mode='L')
        # w.show()
        self.width = rgb.shape[1]
        self.height = rgb.shape[0]
        # depth_vec = depth.flatten()
        return depth

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # If dimensions change they need to be updated in:
    # ThreadPredictor.py:47
    # NetworkVP.py:134 
    def _get_obs(self):
        #if self.image is None:
        #    return None

        # self.current_state = self.image
        #self.current_state = np.concatenate([self.image, self.last_image])
        #self.current_state = self.image
        #self.current_state = (self.current_state.flatten())/128.0 - 1
        # if action is not None:
        #    a = np.array(action).flatten()

        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        pix = np.array((x/256.0,y/144.0)).flatten()
        if self.coord_queue is None:
            self.coord_queue = deque([pix]*self.queue_len)
        else:
            self.coord_queue.append(pix)
            self.coord_queue.popleft() 
        d = np.array([np.linalg.norm(self.t-self.c)])
        self.current_state = np.concatenate(list(self.coord_queue)) # 2x4 = 8
        self.current_state = np.concatenate([np.array(self.v).flatten()/(10.0/self.fps), # 3
                                             #np.array(self.t).flatten()/30.0, # 3
                                             #np.array(self.c).flatten()/30.0, # 3
                                             d/30.0, # 1
                                             self.current_state # 8
                                            ], 0)
        #self.current_state = np.array(d)
        return self.current_state

    def _render(self, mode='human', close=False):
        #self.client.simSetPose(Vector3r(self.c.item(0), self.c.item(1), self.c.item(2)),
        #                       self.client.toQuaternion(math.radians(self.o.item(1)), math.radians(self.o.item(0)),
        #                                                math.radians(self.o.item(2))))

        self.last_image = self.image
        #responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.Scene),
        #                                      ImageRequest(0, AirSimImageType.DepthVis)])
        #if self.episodes % 100 == 0:
        #    if not os.path.exists('./images/episode_' + str(self.episodes) + '/'):
        #        os.makedirs('./images/episode_' + str(self.episodes) + '/')
        #    AirSimClient.write_file(
        #        os.path.normpath('./images/episode_' + str(self.episodes) + '/' + str(self.iteration) + '.png'),
        #        responses[0].image_data_uint8)
        rgb = self._get_rgb(responses[0])
        # response = self.client.simGetImages([ImageRequest(0, AirSimImageType.DepthVis)])[0]
        depth = self._get_depth(responses[1])
        self.image = np.concatenate([rgb, depth], axis=2)
        # self.image = rgb
        if self.last_image is None:
            self.last_image = self.image

        return self.image

    def get_num_actions(self):
        return self.game.env.action_space.n

    ##
    ## End Helper Functions ##
    ##

    def reset(self):
        self.iteration = 0
        self.o = np.matrix([0.0, 0.0, 0.0])
        self.c = np.matrix([0, 0.0, 0.0])
        
        self.t = np.matrix([10.0, 0.0, 0.0])+np.random.normal(0, 5)
        (x, y), _ = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height)) 
        d = np.linalg.norm(self.c-self.t)
        while (x > self.width or x < 0 or y > self.height or y < 0 or d > 30 or d < 5):
            self.t = np.matrix([10.0, 0.0, 0.0])+np.random.normal(0, 5)
            (x,y), _ = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
            d = np.linalg.norm(self.c-self.t)
        self.v = np.matrix([0.0, 0.0, 0.0])
        self.v_t = np.matrix([0.0, 0.0, 0.0])
        self.r = np.matrix([0.0, 0.0, 0.0])
        self.nb_correct = 0
        self.image = None
        #self.fw = None
        self.last_d = d
        #self.c, self.o = self._random_orientation(t)
        #self._render()
        self.start_d = np.linalg.norm(self.t-self.c)
        self.current_state = self._get_obs()
        return self.current_state

    '''
action	 z	 y	 x
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
        x = raw_action % 3 - 1
        raw_action = int(raw_action)/int(3)
        y = raw_action % 3 - 1
        raw_action = int(raw_action)/int(3)
        z = raw_action % 3 - 1

        acc = np.matrix([x,y,z])

        max_v = 10.0/self.fps

        self.v = self.v + acc/self.fps
        if np.linalg.norm(self.v) > max_v: self.v *= max_v/np.linalg.norm(self.v)
        self.c = self.c + self.v
        t_acc = np.random.normal(0, 1)
        if np.linalg.norm(t_acc) > 1: t_acc /= np.linalg.norm(t_acc)
        self.v_t = self.v_t + t_acc/self.fps
        if np.linalg.norm(self.v_t) > max_v: self.v *= max_v/np.linalg.norm(self.v_t)
        #self.t = self.t + self.v_t 
        '''
        self.r = self.r + dR/self.fps

        if self.r.item(0) > max_r: self.r = np.matrix([max_r, self.r.item(1), self.r.item(2)])
        if self.r.item(1) > max_r: self.r = np.matrix([self.r.item(0), max_r, self.r.item(2)])
        if self.r.item(2) > max_r: self.r = np.matrix([self.r.item(0), self.r.item(1), max_r])

        direction = np.dot(rot_mat(self.r.item(0), self.r.item(1), self.r.item(2)), np.transpose(self.c))
        direction = np.transpose(direction)/np.linalg.norm(direction)
        self.o = self.o + self.r
        self.c = self.c + direction*self.v
        if self.c.item(2) > -5:
            self.c = np.matrix([self.c.item(0), self.c.item(1), -5])
        '''
        #self.state = self._render()
        #self.state = self._get_obs()

        self.previous_state = self.current_state
        self.current_state = self._get_obs()
       
        (x, y), target_in_front = projection(self.t, self.c, self.o, w=float(self.width), h=float(self.height))
        d = np.linalg.norm(self.t-self.c)
        
        ## r1
        self.reward = self.last_d - d
        # r2
        #self.reward = 0
        #diff = self.last_d - d
        #if abs(diff) > 0.1:
        #    self.reward = np.sign(diff)
        self.last_d = d
        self.done = (self.iteration > self.max_iter)
        if x > self.width or x < 0 or y > self.height or y < 0 or d > 30 or self.iteration > 500:
        #if d > 30 or self.iteration > 500:
            self.done = True
            self.reward = 0

        if d < 1:
            self.done = True
            self.reward = 100

        #self.reward -= 0.1
        
        self.total_reward += self.reward
        self.iteration += 1

        if self.done is True:
            #if self.episodes % 100 == 0:
                #self.fw.close()
            #    self.fw = None
            #print('Start: '+str(self.start_d)+',End: '+str(self.last_d))
            self.episodes += 1
            self.total_reward = 0
        return self.reward, self.done

