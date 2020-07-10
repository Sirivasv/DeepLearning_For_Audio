from centerFrequency import midi2centerf
import matplotlib.pyplot as plt

plot_notes = []
previous_frequency = 0
for i in range(9):
    current_frequency = i
    
    plot_notes.append((previous_frequency,
        current_frequency-previous_frequency))  
    previous_frequency = current_frequency

fig, ax = plt.subplots()

ax.broken_barh(plot_notes, (0, 5), facecolors='blue', edgecolors="black")
ax.broken_barh([(0, 50)], (5, 5),facecolors='red', edgecolors="black")
ax.set_ylim(0, 35)
ax.set_xlim(0, 8)
ax.set_xlabel('Frequency')
ax.set_yticks([5, 10])
ax.set_yticklabels(['Notes', 'Piano'])
# ax.annotate('C1', (61, 25),
#             xytext=(0.8, 0.9), textcoords='axes fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             fontsize=16,
#             horizontalalignment='right', verticalalignment='top')
plt.show()
