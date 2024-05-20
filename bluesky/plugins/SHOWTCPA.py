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
    for i, pair in enumerate(traf.cd.confpairs):
        # Assuming tcpa array is aligned with confpairs, extract TCPA for the current pair
        # This part needs to be adjusted based on actual data structure and logic in your conflict detection system
        # This is an example; adjust based on your actual data structure
        sorted_pair = tuple(sorted(pair))
        # Skip this pair if it has already been processed
        if sorted_pair in processed_pairs:
            continue
        # Add the sorted pair to the set of processed pairs
        processed_pairs.add(sorted_pair)
        
        tcpa_value = traf.cd.tcpa[i]
        qdr_value = traf.cd.qdr[i]
        distance_value = traf.cd.dist[i]
        dcpa_value = traf.cd.dcpa[i]
        tLOS_value = traf.cd.tLOS[i]

        conflict_info = f"{sorted_pair[0]} - {sorted_pair[1]} TCPA: {tcpa_value:.2f} sec QDR: {qdr_value:.2f} deg Distance: {distance_value:.2f} m DCPA: {dcpa_value:.2f} m tLOS: {tLOS_value:.2f} sec"
        stack.stack(f'ECHO {conflict_info}')
        
