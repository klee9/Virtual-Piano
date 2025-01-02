import time
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
for x in pl.piano_notes[octave+1]:
    notes.append(mixer.Sound(f"/Users/klee9/Desktop/daiv/kirby/notes/{x}.wav"))
channels = [mixer.Channel(i) for i in range(len(notes))]

# active keys to avoid repeated triggering
active_keys = set()

def main():
    while True:
        # play "silent night"
        if keyboard.is_pressed('m'):
            for note, interval in zip(pl.silent_night_notes, pl.silent_night_intervals):
                channels[0].play(notes[pl.keybind[note]])
                time.sleep(interval)

        # play manually
        for key, val in pl.keybind.items():
            if keyboard.is_pressed(key):
                if key not in active_keys:
                    active_keys.add(key)
                    channels[val].play(notes[val], fade_ms=50)
            else:
                active_keys.discard(key)

if __name__ == "__main__":
    main()