from bluesky import stack, traf


def init_plugin():
    config = {
        'plugin_name': 'SHOWTCPA',
        'plugin_type': 'sim'
    }
    stackfunctions = {
        'SHOWTCPA': [
            'SHOWTCPA',
            '',
            show_conflicts_tcpa,
            'Show aircraft pairs in conflict and their TCPA']
    }
    return config, stackfunctions


@stack.command(name='SHOWTCPA')
def show_conflicts_tcpa():
    if not hasattr(traf.cd, 'confpairs') or not len(traf.cd.confpairs):
        stack.stack('ECHO No conflicts detected.')
        return

    # Ensure there's a structure to hold TCPA for each conflict pair
    if not hasattr(traf.cd, 'tcpa') or len(traf.cd.tcpa) == 0:
        stack.stack('ECHO TCPA data not available.')
        return

    msg = "Aircraft Pairs in Conflict and their TCPA (sec):"
    stack.stack(f'ECHO {msg}')

    processed_pairs = set()  # Set to keep track of processed pairs
    # Iterate through detected conflicts
    # Convert frozensets to sorted tuples
    sorted_tuples = [tuple(sorted(pair)) for pair in traf.cd.confpairs_unique]

    # Sort the list of tuples
    sorted_tuples.sort()
    for i, pair in enumerate(sorted_tuples):

        # Add the sorted pair to the set of processed pairs

        tcpa_value = traf.cd.tcpa[i]
        qdr_value = traf.cd.qdr[i]
        distance_value = traf.cd.dist[i]
        dcpa_value = traf.cd.dcpa[i]
        tLOS_value = traf.cd.tLOS[i]

        conflict_info = f"{pair[0]} - {pair[1]} | TCPA: {tcpa_value:.2f} sec | QDR: {qdr_value:.2f} deg | Distance: {distance_value / 1852:.2f} Nautical miles | DCPA: {dcpa_value / 1852:.2f} Nautical miles | tLOS: {tLOS_value:.2f} sec"
        stack.stack(f'ECHO {conflict_info}')
