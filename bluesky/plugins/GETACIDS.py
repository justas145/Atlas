# """ BlueSky plugin template. The text you put here will be visible
#     in BlueSky as the description of your plugin. """
# # Import the global bluesky objects. Uncomment the ones you need
# from bluesky import core, stack, traf  # , settings, navdb, sim, scr, tools


# # Import the necessary modules from BlueSky
from bluesky import stack, traf

# # Initialization function for the plugin


def init_plugin():
    # Configuration parameters
    config = {
        'plugin_name': 'GETACIDS',
        'plugin_type': 'sim'
    }

    stackfunctions = {
        # The command name for your function
        'GETACIDS': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'GETACIDS',
            # A list of the argument types your function accepts. For a description of this, see ...
            'txt',
            # The name of your function in this plugin
            get_aircraft_ids,
            # a longer help text of your function.
            'get all aircraft ids']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


# Function to get each aircraft's information

@stack.command(name='GETACIDS')
def get_aircraft_ids():
    '''Get all aircraft ids'''
    ac_ids_lst = traf.id
    stack.stack("ECHO Aircraft idx: " + str(traf.id))
    for ac in ac_ids_lst:
        stack.stack(ac)

    return ac_ids_lst
