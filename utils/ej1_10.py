from centerFrequency import midi2centerf, diffInCents
import numpy as np

# C2 center frequency
c2_cf = midi2centerf(36)

# C3 center frequency
c3_cf = midi2centerf(48)

center_frequencies_equal_temp = []

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

for i in range(128):
    center_frequencies_equal_temp.append(midi2centerf(i))

center_frequencies_pitagorean = []
center_frequencies_pitagorean.append(c2_cf)
current_freq = c2_cf
for i in range(12):
    current_freq *= 1.5
    while ((current_freq > c3_cf) or (current_freq < c2_cf)):
        if (current_freq > c3_cf):
            current_freq /= 2
        else:
            current_freq *= 2
    center_frequencies_pitagorean.append(current_freq)
center_frequencies_pitagorean = np.array(center_frequencies_pitagorean)
# center_frequencies_pitagorean.sort()
# print(center_frequencies_pitagorean)
for i, c_freq in enumerate(center_frequencies_pitagorean):
    nearest_freq_in_temp = find_nearest(center_frequencies_equal_temp, c_freq)
    dif_in_cents = diffInCents(c_freq, nearest_freq_in_temp)
    sign = "+"
    if dif_in_cents < 0:
        sign = " -"
    print(
        "Semitone #{}: Frequency: {:.2f} Nearest CF: {:.2f} Difference: ".format(
            i,
            c_freq,
            nearest_freq_in_temp
        ) \
        + sign \
        + "{:.2f}".format(
            dif_in_cents
        )
    )