from pyboy import PyBoy

pyboy = PyBoy("pokemon_red.gb", window="SDL2")
pyboy.set_emulation_speed(0)


frame = 0
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

while pyboy.tick():

    # Gets to choosing player's name
    if (frame % 200 == 0 and frame <= 4000) or (frame % 200 == 0 and 7400 < frame <= 8800):
        pyboy.button("a")

    if frame % 50 == 0 and 4099 < frame <= 7350:
        if frame in name_frames:
            pyboy.button(name_frames[frame])

    print(frame)
    frame += 1




pyboy.stop()