import math

def midi2centerf(midi_pitch):
    return (2 ** ((midi_pitch - 69) / 12.0)) * 440.0

def diffInCents(f1, f2):
    return math.log2(f1/f2) * 1200