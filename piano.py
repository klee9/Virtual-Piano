import pygame
import keyboard
import piano_list as pl
from pygame import mixer

# init pygame mixer
pygame.init()
mixer.set_num_channels(100)

# preload sound files
octave = 4
notes = [mixer.Sound(f"/Users/klee9/Desktop/daiv/kirby/notes/{x}.wav") for x in pl.piano_notes[octave]]
channels = [mixer.Channel(i) for i in range(len(notes))]

# active keys to avoid repeated triggering
active_keys = set()

while True:
    for key, val in pl.keybind.items():
        if keyboard.is_pressed(key):
            if key not in active_keys:
                active_keys.add(key)
                channels[val].play(notes[val], fade_ms=50)
        else:
            active_keys.discard(key)