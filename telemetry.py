import fastf1 as f1
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.collections import LineCollection
from matplotlib import cm 
import numpy as np
import pandas as pd

# Cache data so it doesn't need to be loaded each time
f1.Cache.enable_cache('cache')

# Setup plotting
plotting.setup_mpl()

# import data for Belgium 2024

session = f1.get_session(2024, 'Belgium', 'Q')
session.load()

# Get the fastest lap for the two comparisons
fast_norris = session.laps.pick_driver('NOR').pick_fastest()
fast_piastri = session.laps.pick_driver('PIA').pick_fastest()

# Get telemetry data
fastest_nor = fast_norris.get_telemetry().add_distance()
fastest_pia = fast_piastri.get_telemetry().add_distance()

# Create driver column
fastest_nor['Driver'] = 'NOR'
fastest_pia['Driver'] = 'PIA'

telemetry = pd.concat([fastest_nor, fastest_pia], ignore_index=True)

# Create minisectors
num_minisectors = 25

total_distance = max(telemetry['Distance'])

# Generate equally sized minisectors
minisector_length = total_distance / num_minisectors

# Intiatilize the minisector variable
minisectors = [0]

# Add multiples of minisector_length to minisectors
for i in range(0, (num_minisectors - 1)):
    minisectors.append(minisector_length * (i + 1))

telemetry['Minisector'] = telemetry['Distance'].apply(
    lambda dist: (
        int((dist // minisector_length) + 1)
    )
)

# Calculate average speed for each driver by minisector
average_speed = telemetry.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()

# Select driver with highest avg speed
fastest_driver = average_speed.loc[average_speed.groupby(['Minisector'])['Speed'].idxmax()]
# Get rid of speed column, rename driver column
fastest_driver = fastest_driver[['Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_driver'})

# Join fastest driver per minisector with telemetry
telemetry = telemetry.merge(fastest_driver, on=['Minisector'])

# Order data by distance
telemetry = telemetry.sort_values(by=['Distance'])

# Convert driver name to integer
telemetry.loc[telemetry['Fastest_driver'] == 'NOR', 'Fastest_driver_int'] = 1
telemetry.loc[telemetry['Fastest_driver'] == 'PIA', 'Fastest_driver_int'] = 2

x = np.array(telemetry['X'].values)
y = np.array(telemetry['Y'].values)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fastest_driver_array = telemetry['Fastest_driver_int'].to_numpy().astype(float)

cmap = cm.get_cmap('winter', 2)
lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
lc_comp.set_array(fastest_driver_array)
lc_comp.set_linewidth(5)

plt.rcParams['figure.figsize'] = [18, 10]

plt.gca().add_collection(lc_comp)
plt.axis('equal')
plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

cbar = plt.colorbar(mappable=lc_comp, ticks=[1, 2])
cbar.set_ticklabels(['NOR', 'PIA'])

plt.savefig(f"2024_nor_pia_q.png", dpi=300)

plt.show()
