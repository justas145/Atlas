""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """

from random import randint
import numpy as np

# Import the global bluesky objects. Uncomment the ones you need
from bluesky import core, stack, traf  # , settings, navdb, sim, scr, tools


### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    """Plugin initialisation function."""
    # Instantiate our example entity
    example = Example()

    # Configuration parameters
    config = {
        # The name of your plugin
        "plugin_name": "CRASHDETECTION",
        # The type of this plugin. For now, only simulation plugins are possible.
        "plugin_type": "sim",
        "update": example.update,
    }
    stackfunctions = {
        # The command name for your function
        "CRASHDETECTION": [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            "CRASHDETECTION",
            # A list of the argument types your function accepts. For a description of this, see ...
            "txt",
            # The name of your function in this plugin
            Example.update,
            # a longer help text of your function.
            "example",
        ]
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions
    # init_plugin() should always return a configuration dict.


### Entities in BlueSky are objects that are created only once (called singleton)
### which implement some traffic or other simulation functionality.
### To define an entity that ADDS functionality to BlueSky, create a class that
### inherits from bluesky.core.Entity.
### To replace existing functionality in BlueSky, inherit from the class that
### provides the original implementation (see for example the asas/eby plugin).


def log_crash(id1, id2):
    crash_info = f"CRASH: {id1} and {id2}"
    try:
        already_logged = False
        with open("output/crash_log.txt", "r") as file:
            for line in file:
                if crash_info in line:
                    already_logged = True
                    break

        if not already_logged:
            with open("output/crash_log.txt", "a") as file:
                file.write(f"{crash_info}\n")
                print(crash_info)  # Also print to console
    except FileNotFoundError:
        with open("output/crash_log.txt", "w") as file:
            file.write(f"{crash_info}\n")
            print(crash_info)


import math


def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in meters
    R = 6371000

    # Converting latitudes and longitudes from degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    horizontal_distance = R * c
    return horizontal_distance


def calculate_distance(ids, lats, longs, alts):
    distances = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            horizontal_distance = haversine(lats[i], longs[i], lats[j], longs[j])
            alt_diff = (alts[j] - alts[i]) # alts already in meters
            total_distance = math.sqrt(
                horizontal_distance**2 + alt_diff**2
            )  # Pythagorean theorem
            distances.append((ids[i], ids[j], total_distance))
    return distances


class Example(core.Entity):
    """Example new entity object for BlueSky."""

    # Functions that need to be called periodically can be indicated to BlueSky
    # with the timed_function decorator
    @core.timed_function(name="CRASHDETECTION", dt=0.1)
    def update(self):
        """Check if there are any aircraft within 300m of each other."""
        ids = traf.id
        lats = traf.lat
        longs = traf.lon
        alts = traf.alt

        distance = calculate_distance(ids, lats, longs, alts)
        # distance is [(id1, id2, distance), ...]
        # check for crashes. crash is less than 300m
        for id1, id2, distance in distance:
            if distance < 300:
                log_crash(id1, id2)
