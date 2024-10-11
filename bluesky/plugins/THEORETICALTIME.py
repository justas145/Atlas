""" BlueSky plugin to calculate theoretical time to destination. """

import numpy as np
from bluesky import core, stack, traf, sim
from bluesky.tools import geo
import os

theoretical_time_calculator = None

def init_plugin():
    """Plugin initialisation function."""
    global theoretical_time_calculator
    theoretical_time_calculator = TheoreticalTimeCalculator()

    config = {
        "plugin_name": "THEORETICALTIME",
        "plugin_type": "sim",
        "preupdate": theoretical_time_calculator.preupdate,
        "reset": theoretical_time_calculator.reset,
    }

    print("THEORETICALTIME plugin initialized")
    return config

def log_theoretical_time(acid, time):
    global theoretical_time_calculator
    theoretical_time_calculator.log_theoretical_time(acid, time)

class TheoreticalTimeCalculator(core.Entity):
    def __init__(self):
        super().__init__()
        self.calculated = False
        self.log_file = os.path.join("output", "theoretical_time_log.txt")
        print(f"TheoreticalTimeCalculator initialized. Log file: {self.log_file}")

    def preupdate(self):
        if not self.calculated:
            self.calculate()
            self.calculated = True

    def reset(self):
        self.calculated = False
        self.clear_log_file()
        print("TheoreticalTimeCalculator reset called")

    def clear_log_file(self):
        try:
            with open(self.log_file, "w") as file:
                file.write("")  # Clear the file
            print(f"Theoretical time log file cleared: {self.log_file}")
        except Exception as e:
            print(f"Error clearing theoretical time log file: {e}")

    def log_theoretical_time(self, acid, time):
        log_info = f"THEORETICAL TIME: {acid} - Estimated time to destination: {time:.2f} seconds"
        try:
            with open(self.log_file, "a") as file:
                file.write(f"{log_info}\n")
            print(f"Logged to file: {log_info}")
        except Exception as e:
            print(f"Error writing to theoretical time log file: {e}")

    def calculate(self):
        for i in range(traf.ntraf):
            acid = traf.id[i]

            if traf.ap.route[i].dest:
                initial_lat = traf.lat[i]
                initial_lon = traf.lon[i]
                dest_lat = traf.ap.route[i].wplat[-1]
                dest_lon = traf.ap.route[i].wplon[-1]

                distance = geo.kwikdist(initial_lat, initial_lon, dest_lat, dest_lon) * 1852  # Convert nm to meters
                speed = traf.tas[i]  # True airspeed in m/s

                if speed > 0:
                    time_to_destination = distance / speed
                    self.log_theoretical_time(acid, time_to_destination)

@stack.command
def cleartheoreticallog():
    """ Clear the theoretical time log file """
    global theoretical_time_calculator
    if theoretical_time_calculator:
        theoretical_time_calculator.clear_log_file()
    else:
        print("TheoreticalTimeCalculator not initialized")
