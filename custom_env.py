import numpy as np
import os
from colored import fg, bg, attr
import random


class env():
    def __init__(self):
        self.done: bool
        self.playground: np.array
        self.state: int
        self.old_pos: tuple
        self.current_pos: tuple
        self.new_pos: tuple
        self.destination: tuple
        self.state_space: int
        self.current_dst: int
        self.action_space = 4
        self.actions = [(1,0), (-1,0), (0,1), (0,-1)] 
        self.destinations = [(0,0), (3,1), (1,4), (4,4)]
        super().__init__()

    def encode(self,pos,dst_index):
        i = pos[0]
        i *= 5
        i += pos[1]
        i *= 4
        i += dst_index
        return i

    def create(self,dims,start,dst_index):
        assert isinstance(dims, tuple)
        assert isinstance(start, tuple)
        assert isinstance(dst_index, int)

        self.state_space = dims[0]*dims[1]*len(self.destinations)
        self.playground = np.zeros([dims[0], dims[1]])
        self.current_pos = start
        self.destination = self.destinations[dst_index]
        self.current_dst = dst_index
        self.state = self.encode(self.current_pos,self.current_dst)
        self.playground[self.current_pos[0]][self.current_pos[1]] = 1
        self.playground[self.destination[0]][self.destination[1]] = 2

    def reset(self):
        new_dst = random.randint(0,3)
        new_start_pos = (random.randint(0,4), random.randint(0,4))
        self.create((5,5),new_start_pos,new_dst)
        return self.encode(new_start_pos,new_dst)

    
    def rand_action(self):
        rand_act = random.randint(0,3)
        return rand_act

    def step(self, action_index):
        reward: int
        hit_wall: bool
        action = self.actions[action_index]

        self.old_pos = self.current_pos
        self.new_pos = (self.current_pos[0]+action[0],self.current_pos[1]+action[1])

        if self.new_pos[0] <= 4 and self.new_pos[0] >= 0 and self.new_pos[1] <= 4 and self.new_pos[1] >= 0:
            self.current_pos = self.new_pos
            hit_wall = False
        else:
            hit_wall = True

        self.playground[self.old_pos[0]][self.old_pos[1]] = 0
        self.playground[self.current_pos[0]][self.current_pos[1]] = 1

        if self.current_pos == self.destination:
            self.done = True
            reward = 20
        elif hit_wall:
            self.done = False
            reward = -10
        else:
            self.done = False
            reward = -1

        self.state = self.encode(self.current_pos, self.current_dst)
        return self.done, reward, self.state

    def render(self):
        text_list = []
        enviroment = ""
        for y in range(len(self.playground)):
            text_list.append("")
            for x in range(len(self.playground)):
                if self.playground[y][x] == 0:
                    text_list[y] += ": "
                if self.playground[y][x] == 1:
                    color = fg("black") + bg("green")
                    reset = attr("reset")
                    text_list[y] += color + "P " + reset
                if self.playground[y][x] == 2:
                    color = fg("black") + bg("red")
                    reset = attr("reset")
                    text_list[y] += color + "D " + reset

        
        for row in text_list:
            enviroment = enviroment + row + "\n"

        print(enviroment)
        return enviroment
