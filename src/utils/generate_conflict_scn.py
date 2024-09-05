import random
import os
import click

random.seed(42)

@click.command()
@click.option('--conflict-types', default=None, help='Comma-separated list of conflict types (head-on,parallel,t-formation,converging)')
@click.option('--num-conflicts', type=int, required=True, help='Number of conflicts to generate')
@click.option('--aircraft-range', required=True, help='Single number, range (min-max), or comma-separated list of aircraft numbers')
@click.option('--folder-path', type=click.Path(), required=True, help='Output folder path')
@click.option('--use-dh', type=bool, default=True, help='Use altitude difference (dH)')
@click.option('--dh-range', default='625-900', help='Single number, range (min-max), or comma-separated list for dH values')
@click.option('--tlos-hor', default='50-250', help='Single number, range (min-max), or comma-separated list for horizontal TLOS')
@click.option('--tlos-ver', default='30-50', help='Single number, range (min-max), or comma-separated list for vertical TLOS')

def create_conflict_file(conflict_types, num_conflicts, aircraft_range, folder_path, use_dh, dh_range, tlos_hor, tlos_ver):
    # Parse conflict types
    all_conflict_types = ["head-on", "parallel", "t-formation", "converging"]
    if conflict_types:
        conflict_types = [ct.strip() for ct in conflict_types.split(',')]
    else:
        conflict_types = all_conflict_types

    # Parse aircraft range
    aircraft_range = parse_range(aircraft_range)

    # Parse dH range
    dh_values = parse_range(dh_range)

    # Parse TLOS ranges
    tlos_hor_values = parse_range(tlos_hor)
    tlos_ver_values = parse_range(tlos_ver)

    scenarios_per_type = num_conflicts // (len(aircraft_range) * len(conflict_types))
    half_scenarios = scenarios_per_type // 2

    aircraft_types = ["A320", "B737", "A330", "B747", "B777"]

    for num_aircraft in aircraft_range:
        ac_folder = os.path.join(folder_path, f"ac_{num_aircraft}")
        os.makedirs(ac_folder, exist_ok=True)

        for conflict_type in conflict_types:
            if use_dh:
                dH_zero_folder = os.path.join(ac_folder, "no_dH")
                dH_normal_folder = os.path.join(ac_folder, "dH")
                os.makedirs(dH_zero_folder, exist_ok=True)
                os.makedirs(dH_normal_folder, exist_ok=True)
            else:
                scenario_folder = os.path.join(ac_folder, "scenarios")
                os.makedirs(scenario_folder, exist_ok=True)

            for i in range(1, scenarios_per_type + 1):
                filename = f"{conflict_type}_{i}.scn"
                if use_dh:
                    if i <= half_scenarios:
                        dH_list = [0] * num_aircraft
                        file_path = os.path.join(dH_zero_folder, filename)
                    else:
                        dH_list = [random.choice(dh_values) for _ in range(num_aircraft)]
                        file_path = os.path.join(dH_normal_folder, filename)
                else:
                    dH_list = [0] * num_aircraft
                    file_path = os.path.join(scenario_folder, filename)

                with open(file_path, "w") as file:
                    file.write("00:00:00.00>ASAS ON\n")
                    angle_dict = {}

                    lat = random.uniform(-90, 90)
                    long = random.uniform(-180, 180)
                    heading = random.randint(0, 359)
                    flight_level = random.randint(100, 350) * 100
                    speed = random.randint(150, 300)
                    aircraft_type = random.choice(aircraft_types)
                    file.write(
                        f"00:00:00.00>CRE FLIGHT1 {aircraft_type} {lat:.4f} {long:.4f} {heading} {flight_level} {speed}\n"
                    )
                    file.write(f"00:00:00.00>PAN {lat:.4f} {long:.4f}\n")

                    for j in range(2, num_aircraft + 1):
                        target_id = f"FLIGHT{random.randint(1, j-1)}"
                        dpsi = set_dpsi(conflict_type, j, target_id, angle_dict)
                        tlos_hor = random.choice(tlos_hor_values)
                        tlos_ver = random.choice(tlos_ver_values)
                        spd = random.randint(150, 300)
                        aircraft_type = random.choice(aircraft_types)
                        dH = dH_list[j-1]  # Use the pre-generated dH value for this aircraft
                        file.write(
                            f"00:00:00.00>CRECONFS FLIGHT{j} {aircraft_type} {target_id} {dpsi} 0 {tlos_hor} {dH} {tlos_ver} {spd}\n"
                        )

    return f"Generated {num_conflicts} conflict files in {folder_path} with aircraft ranging from {min(aircraft_range)} to {max(aircraft_range)}."

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    elif ',' in range_str:
        return [int(x.strip()) for x in range_str.split(',')]
    else:
        return [int(range_str)]

def set_dpsi(conflict_type, aircraft_index, target_id, angle_dict):
    if conflict_type == "t-formation":
        possible_angles = [90, 270]
        used_angles = angle_dict.get(target_id, [])
        if len(used_angles) == 1:
            dpsi = 270 if used_angles[0] == 90 else 90
        else:
            dpsi = random.choice(possible_angles)
        angle_dict.setdefault(target_id, []).append(dpsi)
        return dpsi
    elif conflict_type == "head-on":
        return 180
    elif conflict_type == "parallel":
        return 0
    else:  # converging
        return random.choice([x for x in range(360) if not (x in [0, 90, 180, 270])])

if __name__ == '__main__':
    create_conflict_file()
