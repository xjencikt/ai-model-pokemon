from pyboy import PyBoy
from sympy.physics.units import action
import numpy as np
import random
import threading

frame = 1

# Writes my player's name (TJ) and Gary's name (KAZ)
name_frames = {
    4900: "right", 4950: "down", 5000: "down", 5050: "a", 5100: "left", 5150: "up", 5200: "a",
    5250: "right", 5300: "right", 5350: "right", 5400: "right", 5450: "right", 5500: "right",
    5550: "right", 5600: "right", 5650: "down", 5700: "down", 5750: "down", 5800: "a", 6200: "a",
    6400: "a", 6600: "a", 6800: "a", 6900: "a", 7000: "a", 7100: "a", 7200: "a",7250: "right",
    7300: "down", 7350: "a", 7400: "left", 7450: "up", 7500: "a", 7550: "right", 7600: "right",
    7650: "right", 7700: "right", 7750: "right", 7800: "right", 7850: "right", 7900: "down",
    7950: "down", 8000: "a", 8050: "down", 8100: "down", 8150: "right", 8200: "a",
}


CLICK_ACTIONS = {
    0: "noop",
    1: "a",
    2: "b",
}

MOVE_ACTIONS = {
    3: "up",
    4: "down",
    5: "left",
    6: "right",
}

training_running = False

class PokemonRed:
    def __init__(self, name, sound, sound_emulated, window="SDL2"):
        self.name = name
        self.window = window
        self.frame = 0

        self.pyboy = PyBoy(self.name, window=self.window, sound_volume=sound, sound_emulated=sound_emulated)
        self.pyboy.set_emulation_speed(0)

        self.map_id = self.pyboy.memory[0xD35E]
        self.in_battle_state = self.pyboy.memory[0xD057]

    def create_game(self):
        while self.pyboy.tick():

            x = self.pyboy.memory[0xD361]
            y = self.pyboy.memory[0xD362]

            # player_x = self.pyboy.memory[0xD361]
            # player_y = self.pyboy.memory[0xD362]
            # map_id = self.pyboy.memory[0xD35E]

            # Gets to choosing player's name
            if self.frame % 200 == 0 and 800 < self.frame < 1300:
                self.pyboy.button("down", 3)

            elif self.frame == 1400:
                self.pyboy.button("up", 3)

            elif ((self.frame % 200 == 0 and self.frame <= 800) or (self.frame % 200 == 0 and 1700 < self.frame <= 4800)
                    or (self.frame % 200 == 0 and 8250 < self.frame <= 9999)):
                self.pyboy.button("a", 3)

            elif self.frame % 50 == 0 and 4800 < self.frame <= 8250:
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

            if self.frame % 100 == 0 and 1200 < self.frame < 2000:
                self.pyboy.button("a", 3)

            if self.frame == 2200:
                with open("saves/pokemon_red_save.state", "wb") as f:
                    self.pyboy.save_state(f)
                break

            self.frame += 1

    def end_game(self):
        self.pyboy.stop()

    def load_game(self, save):
        with open(save, "rb") as f:
            self.pyboy.load_state(f)

    def get_position(self):
        x = self.pyboy.memory[0xD361]
        y = self.pyboy.memory[0xD362]

        return [x, y]

    def get_state(self):
        ram = self.pyboy.memory
        x = ram[0xD361]
        y = ram[0xD362]
        map_id = ram[0xD35E]

        tile = self.pyboy.tilemap_background
        # Radius 3x3
        radius_tiles = []
        for px in range(-1, 2):
            for py in range(-1, 2):
                radius_tile = tile[x + px][y + py] / 512
                radius_tiles.append(radius_tile)


        return np.array([x / 255.0, y / 255.0, map_id / 255.0] + radius_tiles, dtype=np.float32)

    def player_action(self, act_idx):
        if act_idx > 2:
                button = MOVE_ACTIONS.get(act_idx)
                self.pyboy.button_press(button) # DO NOT USE pyboy.button!!! - use press/release
                for _ in range(10):
                    self.pyboy.tick()
                self.pyboy.button_release(button)
        else:
            if act_idx != 0:
                button = CLICK_ACTIONS.get(act_idx)
                self.pyboy.button_press(button) # DO NOT USE pyboy.button!!! - use press/release
                for _ in range(10):
                    self.pyboy.tick()
                self.pyboy.button_release(button)



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


