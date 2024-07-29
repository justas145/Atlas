import random
import os


def create_conflict_file(num_conflicts, aircraft_range, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    aircraft_types = ["A320", "B737", "A330", "B747", "B777"]
    conflict_types = ["head-on", "parallel", "t-formation", "converging"]
    num_each = num_conflicts // (len(aircraft_range) * len(conflict_types))

    for num_aircraft in aircraft_range:
        for conflict_type in conflict_types:
            for _ in range(num_each):
                filename = f"conflict_{conflict_type}_{num_aircraft}.scn"
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "w") as file:
                    # Write the constant first line
                    file.write("00:00:00.00>ASAS ON\n")
                    
                    angle_dict = {}

                    # Generate and write the CRE line for FLIGHT1
                    lat = random.uniform(-90, 90)
                    long = random.uniform(-180, 180)
                    heading = random.randint(0, 359)
                    flight_level = random.randint(100, 350) * 100
                    speed = random.randint(150, 300)
                    aircraft_type = random.choice(aircraft_types)
                    file.write(
                        f"00:00:00.00>CRE FLIGHT1 {aircraft_type} {lat:.4f} {long:.4f} {heading} {flight_level} {speed}\n"
                    )

                    # Write the PAN line
                    file.write(f"00:00:00.00>PAN {lat:.4f} {long:.4f}\n")

                    # Generate additional CRECONFS lines
                    for i in range(2, num_aircraft + 1):
                        target_id = f"FLIGHT{random.randint(1, i-1)}"
                        if conflict_type == "head-on":
                            dpsi = 180
                        elif conflict_type == "parallel":
                            dpsi = 0
                            speed -= 50  # Reduce speed for trailing aircraft
                        elif conflict_type == "t-formation":
                            possible_angles = [90, 270]
                            used_angles = angle_dict.get(target_id, [])
                            if len(used_angles) == 1:
                                dpsi = 270 if used_angles[0] == 90 else 90
                            else:
                                dpsi = random.choice(possible_angles)
                            angle_dict.setdefault(target_id, []).append(dpsi)

                        else:  # converging or other cases
                            dpsi = random.choice(
                                [x for x in range(360) if not (x in [0, 90, 180, 270])]
                            )

                        cpa = 0  # Constant zero as per specs
                        tlos_hor = random.randint(50, 250)
                        dH = random.randint(625, 900)
                        tlos_ver = random.randint(30, 50)
                        spd = random.randint(150, 300)
                        aircraft_type = random.choice(aircraft_types)
                        print(conflict_type, dpsi)
                        file.write(
                            f"00:00:00.00>CRECONFS FLIGHT{i} {aircraft_type} {target_id} {dpsi} {cpa} {tlos_hor} {dH} {tlos_ver} {spd}\n"
                        )

    return f"Generated {num_conflicts} conflict files in {folder_path} with aircraft ranging from {aircraft_range[0]} to {aircraft_range[-1]}."


# Example usage
create_conflict_file(120, [2, 3, 4], "../../bluesky/scenario/TEST/Big")
