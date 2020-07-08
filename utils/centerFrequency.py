def midi2centerf(midi_pitch):
    return (2 ** ((midi_pitch - 69) / 12.0)) * 440.0