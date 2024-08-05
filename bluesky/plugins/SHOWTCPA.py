import math
from bluesky import stack, traf


def init_plugin():
    config = {"plugin_name": "SHOWTCPA", "plugin_type": "sim"}
    stackfunctions = {
        "SHOWTCPA": [
            "SHOWTCPA",
            "",
            show_conflicts_tcpa,
            "Show aircraft pairs in conflict and their TCPA",
        ]
    }
    return config, stackfunctions


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


@stack.command(name="SHOWTCPA")
def show_conflicts_tcpa():
    if not hasattr(traf.cd, "confpairs") or not len(traf.cd.confpairs):
        stack.stack("ECHO No conflicts detected.")
        return

    # Ensure there's a structure to hold TCPA for each conflict pair
    if not hasattr(traf.cd, "tcpa") or len(traf.cd.tcpa) == 0:
        stack.stack("ECHO TCPA data not available.")
        return

    msg = "Aircraft Pairs in Conflict and their TCPA (sec):"
    stack.stack(f"ECHO {msg}")

    processed_pairs = set()  # Set to keep track of processed pairs
    involved_aircraft = set()  # Set to keep track of unique aircraft in conflict

    # Iterate through detected conflicts
    # Convert frozensets to sorted tuples
    sorted_tuples = [tuple(sorted(pair)) for pair in traf.cd.confpairs_unique]

    # Sort the list of tuples
    sorted_tuples.sort()
    for i, pair in enumerate(sorted_tuples):

        tcpa_value = traf.cd.tcpa[i]
        qdr_value = traf.cd.qdr[i]
        dcpa_value = traf.cd.dcpa[i]
        tLOS_value = traf.cd.tLOS[i]

        ids = traf.id
        lats = traf.lat
        longs = traf.lon
        alts = traf.alt
        vs = traf.vs

        # Indices of the pair in the aircraft list
        index_0 = ids.index(pair[0])
        index_1 = ids.index(pair[1])

        # Calculate horizontal distance using haversine function
        horizontal_distance_m = haversine(
            lats[index_0], longs[index_0], lats[index_1], longs[index_1]
        )
        horizontal_distance_nm = (
            horizontal_distance_m / 1852
        )  # Convert meters to nautical miles

        # Calculate vertical distance
        vertical_distance_ft = (
            abs(alts[index_0] - alts[index_1]) * 3.28084
        )  # Convert meters to feet


        # Record involved aircraft
        involved_aircraft.update(pair)

        # Prepare conflict information
        conflict_info = (
            f"{pair[0]} - {pair[1]} | "
            f"TCPA: {tcpa_value:.2f} sec | "
            f"QDR: {qdr_value:.2f} deg | "
            f"Distance: {horizontal_distance_nm:.2f} Nautical miles | "
            f"Vertical Separation: {vertical_distance_ft:.2f} ft | "
            f"Horizontal Distance: {horizontal_distance_nm:.2f} Nautical miles | "
            f"DCPA: {dcpa_value / 1852:.2f} Nautical miles | "
            f"tLOS: {tLOS_value:.2f} sec"
        )
        stack.stack(f"ECHO {conflict_info}")

    num_ac_conf = len(involved_aircraft)
    stack.stack(f"ECHO Number of aircraft in conflict: {num_ac_conf}")
    # Display altitude information for each unique aircraft in conflict
    stack.stack("ECHO Aircraft Altitude Information:")
    for aircraft in involved_aircraft:
        index = ids.index(aircraft)
        altitude_ft = alts[index] * 3.28084
        vs_status = (
            "level"
            if round(float(vs[index]), 1) == 0.0
            else ("ascending" if vs[index] > 0 else "descending")
        )
        stack.stack(
            f"ECHO Aircraft {aircraft}: Altitude {altitude_ft:.2f} ft ({vs_status})"
        )
