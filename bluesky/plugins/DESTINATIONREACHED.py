""" BlueSky plugin to log when aircraft reach their destinations. """

import numpy as np
from bluesky import core, stack, traf, sim
from bluesky.tools import geo
import os

# Global variable to store the logger instance
destination_logger = None

def init_plugin():
    """Plugin initialisation function."""
    global destination_logger
    destination_logger = DestinationLogger()

    config = {
        "plugin_name": "DESTINATIONREACHED",
        "plugin_type": "sim",
        "update": destination_logger.update,
        "reset": destination_logger.reset,
    }

    print("DESTINATIONREACHED plugin initialized")
    return config

def log_destination_reached(acid, time_taken):
    global destination_logger
    destination_logger.log_destination_reached(acid, time_taken)

class DestinationLogger(core.Entity):
    def __init__(self):
        super().__init__()
        self.start_times = {}
        self.last_positions = {}
        self.log_file = os.path.join("output", "destination_log.txt")
        print(f"DestinationLogger initialized. Log file: {self.log_file}")

    def reset(self):
        """Reset the plugin state and clear the log file."""
        self.start_times.clear()
        self.last_positions.clear()
        self.clear_log_file()
        print("DestinationLogger reset called")

    def clear_log_file(self):
        try:
            with open(self.log_file, "w") as file:
                file.write("")  # Clear the file
            print(f"Destination log file cleared: {self.log_file}")
        except Exception as e:
            print(f"Error clearing destination log file: {e}")

    def log_destination_reached(self, acid, time_taken):
        log_info = f"DESTINATION REACHED: {acid} - Time taken: {time_taken:.2f} seconds"
        try:
            with open(self.log_file, "a") as file:
                file.write(f"{log_info}\n")
            print(f"Logged to file: {log_info}")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    @core.timed_function(name="DESTINATIONREACHED", dt=0.1)
    def update(self):
        for i in range(traf.ntraf):
            acid = traf.id[i]
            
            if traf.ap.route[i].dest and len(traf.ap.route[i].wplat) > 0:
                current_pos = (traf.lat[i], traf.lon[i])
                
                if acid not in self.start_times:
                    self.start_times[acid] = sim.simt
                    self.last_positions[acid] = current_pos
                


                dest_lat = traf.ap.route[i].wplat[-1]
                dest_lon = traf.ap.route[i].wplon[-1]
                dist_to_dest = geo.kwikdist(traf.lat[i], traf.lon[i], dest_lat, dest_lon)
                print(dist_to_dest)
                if dist_to_dest < 0.1:  # Within 1 NM of destination
                    time_taken = sim.simt - self.start_times[acid]
                    print(time_taken)
                    if time_taken > 0:
                        self.log_destination_reached(acid, time_taken)
                    del self.start_times[acid]
                    del self.last_positions[acid]

@stack.command
def clearlog():
    """ Clear the destination log file """
    global destination_logger
    if destination_logger:
        destination_logger.clear_log_file()
    else:
        print("DestinationLogger not initialized")
