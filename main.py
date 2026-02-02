from pyboy import PyBoy
from sympy.physics.units import action
import numpy as np


frame = 1

# Writes my player's name (TJ) and Gary's name (KAZ)
name_frames = {
    4100: "right", 4150: "down", 4200: "down", 4250: "a", 4300: "left", 4350: "up", 4400: "a",
    4450: "right", 4500: "right", 4550: "right", 4600: "right", 4650: "right", 4700: "right",
    4750: "right", 4800: "right", 4850: "down", 4900: "down", 4950: "down", 5000: "a", 5100: "a",
    5200: "a", 5300: "a", 5500: "a", 5700: "a", 5900: "a", 6100: "a", 6300: "a",6350: "right",
    6400: "down", 6450: "a", 6500: "left", 6550: "up", 6600: "a", 6650: "right", 6700: "right",
    6750: "right", 6800: "right", 6850: "right", 6900: "right", 6950: "right", 7000: "down",
    7050: "down", 7100: "a", 7150: "down", 7200: "down", 7250: "right", 7300: "a",
}

ACTIONS = ["noop", "a", "b",
           "up", "down",
           "right", "left",
           ]


class PokemonRed:
    def __init__(self, name, window="SDL2"):
        self.name = name
        self.window = window
        self.frame = 0

        self.pyboy = PyBoy(self.name, window=self.window)
        self.pyboy.set_emulation_speed(0)

        self.map_id = self.pyboy.memory[0xD35E]
        self.in_battle_state = self.pyboy.memory[0xD057]

    def _position(self):
        self.x = self.pyboy.memory[0xD361]
        self.y = self.pyboy.memory[0xD362]

        return [self.x, self.y]


    def create_game(self):

        while self.pyboy.tick():

            # Gets to choosing player's name
            if (self.frame % 200 == 0 and self.frame <= 4000) or (self.frame % 200 == 0 and 7400 < self.frame <= 9999):
                self.pyboy.button("a", 3)

            if self.frame % 50 == 0 and 4099 < self.frame <= 7350:
                if self.frame in name_frames:
                    self.pyboy.button(name_frames[self.frame])

            if self.frame == 10000:
                 break

            self.frame += 1


    def save_game(self):
        self.frame = 0
        while self.pyboy.tick():
            if self.frame == 400:
                self.pyboy.button("start", 3)

            if self.frame % 100 == 0 and 700 < self.frame < 1100:
                self.pyboy.button("down", 3)

            if self.frame % 100 == 0 and 1200 < self.frame < 1800:
                self.pyboy.button("a", 3)

            if self.frame == 2000:
                with open("pokemon_red_save.state", "wb") as f:
                    self.pyboy.save_state(f)
                break


            print(self.frame)
            self.frame += 1

    def end_game(self):
        self.pyboy.stop()

    def load_game(self, save):
        with open(save, "rb") as f:
            self.pyboy.load_state(f)

    def get_state(self):
        ram = self.pyboy.memory
        x = ram[0xD361]
        y = ram[0xD362]
        map_id = ram[0xD35E]
        return np.array([x, y, map_id], dtype=np.float32)

    def player_action(self, act_idx):
        if act_idx != 0:
            button = ACTIONS[act_idx]
            print(button)
            self.pyboy.button(button)
        else:
            pass

pokemon_red = PokemonRed("pokemon_red.gb", window="SDL2")

# Creating First save
# pokemon_red.create_game()
# pokemon_red.save_game()
# pokemon_red.end_game()
#pokemon_red.load_game()
#
# player_x = self.pyboy.memory[0xD361]
# player_y = self.pyboy.memory[0xD362]
# map_id = self.pyboy.memory[0xD35E]
# in_battle = self.pyboy.memory[0xD057]


