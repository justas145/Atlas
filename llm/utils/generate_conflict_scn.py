import random
import os

random.seed(42)

def create_conflict_file(num_conflicts, aircraft_range, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    aircraft_types = ["A320", "B737", "A330", "B747", "B777"]
    conflict_types = ["head-on", "parallel", "t-formation", "converging"]
    scenarios_per_type = num_conflicts // (len(aircraft_range) * len(conflict_types))
    half_scenarios = scenarios_per_type // 2  # Half with dH = 0

    for num_aircraft in aircraft_range:
        ac_folder = os.path.join(folder_path, f"ac_{num_aircraft}")
        os.makedirs(ac_folder, exist_ok=True)

        for conflict_type in conflict_types:
            dH_zero_folder = os.path.join(ac_folder, "no_dH")
            dH_normal_folder = os.path.join(ac_folder, "dH")
            os.makedirs(dH_zero_folder, exist_ok=True)
            os.makedirs(dH_normal_folder, exist_ok=True)

            for i in range(1, scenarios_per_type + 1):
                filename = f"{conflict_type}_{i}.scn"
                if i <= half_scenarios:
                    dH = 0
                    file_path = os.path.join(dH_zero_folder, filename)
                else:
                    dH = random.randint(625, 900)
                    file_path = os.path.join(dH_normal_folder, filename)

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
                        tlos_hor = random.randint(50, 250)
                        tlos_ver = random.randint(30, 50)
                        spd = random.randint(150, 300)
                        aircraft_type = random.choice(aircraft_types)
                        file.write(
                            f"00:00:00.00>CRECONFS FLIGHT{j} {aircraft_type} {target_id} {dpsi} 0 {tlos_hor} {dH} {tlos_ver} {spd}\n"
                        )

    return f"Generated {num_conflicts} conflict files in {folder_path} with aircraft ranging from {aircraft_range[0]} to {aircraft_range[-1]}."


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


# Example usage
create_conflict_file(120, [2, 3, 4], "../../bluesky/scenario/TEST/Big")
