from centerFrequency import midi2centerf, diffInCents
import numpy as np
import math

center_frequencies = []

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

for i in range(128):
    center_frequencies.append(midi2centerf(i))

center_frequencies = np.array(center_frequencies)

initial_midi_pitch = 36 # C2
initial_frequency = midi2centerf(36)

for i in range(1, 17):
    current_frequecy = initial_frequency * i
    nearest_center_frequency = find_nearest(
        center_frequencies, current_frequecy)
    difference_in_cents = diffInCents(
        current_frequecy, nearest_center_frequency)
    
    difference_in_cents_opt = (math.log2(i) - (1.0/12.0) * 1200)
    sign = ""
    if current_frequecy > nearest_center_frequency:
        sign = " +"
    print(
        "Harmonic #{}: Frequency: {:.2f} Nearest CF: {:.2f} Difference: ".format(
            i,
            current_frequecy,
            nearest_center_frequency
        ) \
        + sign \
        + "{:.2f}".format(
            difference_in_cents_opt
        )
    )