import pandas as pd
import subprocess
import os
import click
import sys

@click.command()
@click.option('--csv_file', required=True, help='Path to the CSV file containing the scenarios')
def rerun_skipped_scenarios(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter rows where log is 'skipped'
    skipped_rows = df[df['json_results'].str.contains('Error code: 429', na=False)]
    
    # Remove skipped rows from the original DataFrame
    df = df[df['log'] != 'skipped']
    
    # Rerun skipped scenarios
    for _, row in skipped_rows.iterrows():
        model_name = row['model_name']
        temperature = row['temperature']
        scenario = row['scenario']
        
        # ensure scenario slashes are correct, instead of \ use /
        scenario = scenario.replace("\\", "/")
        # ensure there is .scn at the end of the scenario
        if not scenario.endswith(".scn"):
            scenario += ".scn"
        
        # Construct the command to run conflict_main.py
        venv_python = sys.executable
        command = [
            venv_python, "conflict_main.py",
            "--model_name", model_name,
            "--temperature", str(temperature),
            "--scenario", scenario,
            "--output_csv", os.path.splitext(csv_file)[0] + '_2.csv'  # Append results to a new CSV file with '_rerun_skipped' suffix
        ]
        
        print(f"Rerunning scenario: {scenario}")
        print(f"Command: {' '.join(command)}")
        
        # Run the command
        subprocess.run(command, check=True)
        
        # # Check if the rerun was successful
        # updated_df = pd.read_csv(csv_file)
        # successful_run = updated_df[
        #     (updated_df['model_name'] == model_name) &
        #     (updated_df['temperature'] == temperature) &
        #     (updated_df['scenario'] == scenario) &
        #     (updated_df['log'] != 'skipped')
        # ]
        
        # if not successful_run.empty:
        #     # Remove the skipped row if a successful run was found
        #     df = df[~((df['model_name'] == model_name) &
        #               (df['temperature'] == temperature) &
        #               (df['scenario'] == scenario) &
        #               (df['log'] == 'skipped'))]
        #     print(f"Successfully reran scenario: {scenario}")
        # else:
        #     print(f"Failed to rerun scenario: {scenario}")
    
    
    print("Finished rerunning skipped scenarios and updating the CSV file.")

if __name__ == "__main__":
    rerun_skipped_scenarios()
