# %%
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
# Set default font and plot styles
matplotlib.rc("font", size=12, family="Ubuntu")
matplotlib.rc("lines", linewidth=2, markersize=8)
matplotlib.rc("grid", color="darkgray", linestyle=":")


# %%


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    try:
        df["temperature"] = df["temperature"].str.replace("p", ".").astype(float)
    except AttributeError:
        print("Temperature column is already float")

    # replace model names instead of _ with -
    df["model_name"] = df["model_name"].str.replace("_", "-")

    # if gpt-4o is in model name then replace it with GPT-4o entirely
    # if llama is in model name then replace it with Llama-70B entirely
    # Custom function to replace specific model names
    def replace_model_name(model_name):
        if "gpt-4o" in model_name:
            return "GPT-4o"
        elif "llama" in model_name:
            return "Llama-70B"
        return model_name

    # Apply the custom function to the model_name column
    df["model_name"] = df["model_name"].apply(replace_model_name)

    return df


def calculate_scores(df):
    # Total score by model, agent type, experience library, and num_aircraft
    total_scores = (
        df.groupby(
            [
                "model_name",
                "agent_type",
                "experience_library",
                "num_aircraft",
                "conflict_type",
                "conflict_with_dH",
            ]
        )
        .score.sum()
        .reset_index()
    )

    # Average score by model, agent type, experience library, and num_aircraft
    average_scores = (
        df.groupby(
            [
                "model_name",
                "agent_type",
                "experience_library",
                "num_aircraft",
                "conflict_type",
                "conflict_with_dH",
            ]
        )
        .score.mean()
        .reset_index()
    )

    average_num_tools_used = (
        df.groupby(
            [
                "model_name",
                "agent_type",
                "experience_library",
                "num_aircraft",
                "conflict_type",
            ]
        )
        .num_total_commands.mean()
        .reset_index()
    )

    average_num_commands_sent = (
        df.groupby(
            [
                "model_name",
                "agent_type",
                "experience_library",
                "num_aircraft",
                "conflict_type",
            ]
        )
        .num_send_commands.mean()
        .reset_index()
    )

    # Success rate calculations
    # , "num_aircraft"
    success_counts = (
        df[df["score"] == 1]
        .groupby(["model_name", "agent_type", "experience_library"])
        .size()
        .reset_index(name="success_count")
    )
    total_counts = (
        df.groupby(["model_name", "agent_type", "experience_library"])
        .size()
        .reset_index(name="total_count")
    )

    # Success rate calculations without num_aircraft
    success_counts = (
        df[df["score"] == 1]
        .groupby(["model_name", "agent_type", "experience_library"])
        .size()
        .reset_index(name="success_count")
    )
    total_counts = (
        df.groupby(["model_name", "agent_type", "experience_library"])
        .size()
        .reset_index(name="total_count")
    )
    success_rate = pd.merge(
        success_counts,
        total_counts,
        on=["model_name", "agent_type", "experience_library"],
        how="outer",  # Use outer join to include all combinations
    ).fillna(0)  # Fill NaN values with 0 for success_count
    success_rate["success_rate"] = (
        success_rate["success_count"] / success_rate["total_count"]
    )

    # Success rate calculations with num_aircraft
    success_counts_group_ac = (
        df[df["score"] == 1]
        .groupby(["model_name", "agent_type", "experience_library", "num_aircraft"])
        .size()
        .reset_index(name="success_count")
    )
    total_counts_group_ac = (
        df.groupby(["model_name", "agent_type", "experience_library", "num_aircraft"])
        .size()
        .reset_index(name="total_count")
    )
    success_rate_group_ac = pd.merge(
        success_counts_group_ac,
        total_counts_group_ac,
        on=["model_name", "agent_type", "experience_library", "num_aircraft"],
        how="outer",  # Use outer join to include all combinations
    ).fillna(0)  # Fill NaN values with 0 for success_count

    success_rate_group_ac["success_rate"] = (
        success_rate_group_ac["success_count"] / success_rate_group_ac["total_count"]
    )
    # Success rate calculations with conflict_type
    success_counts_group_ct = (
        df[df["score"] == 1]
        .groupby(["model_name", "agent_type", "experience_library", "conflict_type"])
        .size()
        .reset_index(name="success_count")
    )
    total_counts_group_ct = (
        df.groupby(["model_name", "agent_type", "experience_library", "conflict_type"])
        .size()
        .reset_index(name="total_count")
    )
    success_rate_group_ct = pd.merge(
        success_counts_group_ct,
        total_counts_group_ct,
        on=["model_name", "agent_type", "experience_library", "conflict_type"],
        how="outer",  # Use outer join to include all combinations
    ).fillna(0)  # Fill NaN values with 0 for success_count

    success_rate_group_ct["success_rate"] = (
        success_rate_group_ct["success_count"] / success_rate_group_ct["total_count"]
    )

    # Success rate calculations with dH
    success_counts_group_dh = (
        df[df["score"] == 1]
        .groupby(["model_name", "agent_type", "experience_library", "conflict_with_dH"])
        .size()
        .reset_index(name="success_count")
    )
    total_counts_group_dh = (
        df.groupby(
            ["model_name", "agent_type", "experience_library", "conflict_with_dH"]
        )
        .size()
        .reset_index(name="total_count")
    )
    success_rate_group_dh = pd.merge(
        success_counts_group_dh,
        total_counts_group_dh,
        on=["model_name", "agent_type", "experience_library", "conflict_with_dH"],
        how="outer",  # Use outer join to include all combinations
    ).fillna(0)  # Fill NaN values with 0 for success_count

    success_rate_group_dh["success_rate"] = (
        success_rate_group_dh["success_count"] / success_rate_group_dh["total_count"]
    )

    return (
        total_scores,
        average_scores,
        success_rate,
        success_rate_group_ac,
        success_rate_group_ct,
        success_rate_group_dh,
        average_num_tools_used,
        average_num_commands_sent,
    )


df = load_and_prepare_data("../results/FINAL_V5-2.csv")

(
    total_scores,
    average_scores,
    success_rate,
    success_rate_group_ac,
    success_rate_group_ct,
    success_rate_group_dh,
    average_num_tools_used,
    average_num_commands_sent,
) = calculate_scores(df)


data = success_rate.eval("success_rate = success_rate * 100").sort_values(
    "agent_type", ascending=False
)

data["hue_label"] = data["agent_type"].str.replace("_", " ").str.title() + data[
    "experience_library"
].apply(lambda x: " + Experience" if x else "")


plt.figure(figsize=(6, 4))
# Create a formatted label for better legend readability


colors = ["lightblue", "tab:blue", "salmon", "tab:red"] * 2

sns.barplot(
    x="model_name",
    y="success_rate",
    hue="hue_label",
    data=data,
    palette=colors,
    width=0.9,
    gap=0.2,
    # palette="husl",
)
plt.xlabel(None)
plt.ylabel(None)
# plt.legend(framealpha=0.15, title="Agent Type/Experience")
plt.legend(
    bbox_to_anchor=(0.48, 1.3),
    loc="upper center",
    ncol=2,
    frameon=False,
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))

# plt.xticks(rotation=45)
# plt.grid(True)
plt.tight_layout()

plt.savefig("../results/figures2/bar_chart_success_rate.pdf", bbox_inches="tight")

# %%

data = success_rate_group_ac.eval("success_rate = success_rate * 100")

models = data["model_name"].unique()

fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)


for i, model in enumerate(models):
    model_data = data[data["model_name"] == model].copy()

    model_data["hue_label"] = model_data["agent_type"].str.replace(
        "_", " "
    ).str.title() + model_data["experience_library"].apply(
        lambda x: " + Exp" if x else ""
    )

    model_data = model_data.sort_values("agent_type", ascending=False)

    ax = axes[i]

    colors = ["lightblue", "tab:blue", "salmon", "tab:red"]

    sns.lineplot(
        x="num_aircraft",
        y="success_rate",
        hue="hue_label",
        data=model_data,
        marker="o",
        palette=colors,
        errorbar=None,
        ax=ax,
    )

    ax.text(
        0.95,
        0.03,
        model,
        fontsize=20,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xticks([2, 3, 4])

    ax.set_xlabel("Number of Aircraft")
    ax.set_ylabel(None)
    ax.legend(framealpha=1, facecolor="white")
    ax.grid()

    # Ensure that the y-axis starts at zero
    ax.set_ylim(0, 110)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}%"))

    tag = model.replace(":", "_").replace(" ", "_")

plt.tight_layout()

plt.savefig(f"../results/figures2/success_rate_by_aircraft_number.pdf", bbox_inches="tight")

# ... existing code ...


def plot_success_rate_by_conflict_type_bar(success_rate_group_ct):
    models = success_rate_group_ct["model_name"].unique()

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for i, model in enumerate(models):
        model_data = success_rate_group_ct[
            success_rate_group_ct["model_name"] == model
        ].copy()

        model_data["hue_label"] = model_data["agent_type"].str.replace(
            "_", " "
        ).str.title() + model_data["experience_library"].apply(
            lambda x: " + Exp" if x else ""
        )

        model_data = model_data.sort_values("agent_type", ascending=False)

        ax = axes[i]

        colors = ["lightblue", "tab:blue", "salmon", "tab:red"]

        sns.barplot(
            x="conflict_type",
            y="success_rate",
            hue="hue_label",
            data=model_data,
            palette=colors,
            errorbar=None,
            ax=ax,
        )

        ax.text(
            0.95,
            0.03,
            model,
            fontsize=20,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.set_xlabel("Conflict Type")
        ax.set_ylabel(None)
        ax.legend(framealpha=0.1, facecolor="white")

        # Ensure that the y-axis starts at zero
        ax.set_ylim(0, 1.1)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x*100)}%"))

    plt.tight_layout()

    plt.savefig(
        "../results/figures2/success_rate_by_conflict_type_bar.pdf", bbox_inches="tight"
    )


# Call the function with your data
plot_success_rate_by_conflict_type_bar(success_rate_group_ct)


def plot_success_rate_by_conflict_with_dh_bar(success_rate_group_dh):
    models = success_rate_group_dh["model_name"].unique()

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for i, model in enumerate(models):
        model_data = success_rate_group_dh[
            success_rate_group_dh["model_name"] == model
        ].copy()

        model_data["hue_label"] = model_data["agent_type"].str.replace(
            "_", " "
        ).str.title() + model_data["experience_library"].apply(
            lambda x: " + Exp" if x else ""
        )

        model_data = model_data.sort_values("agent_type", ascending=False)

        ax = axes[i]

        colors = ["lightblue", "tab:blue", "salmon", "tab:red"]

        sns.barplot(
            x="conflict_with_dH",
            y="success_rate",
            hue="hue_label",
            data=model_data,
            palette=colors,
            errorbar=None,
            ax=ax,
        )

        ax.text(
            0.95,
            0.03,
            model,
            fontsize=20,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="none", edgecolor="none", pad=0),
        )

        ax.set_xlabel("Vertical Conflict")
        ax.set_ylabel(None)
        ax.legend(framealpha=0.1, facecolor="white")

        # Ensure that the y-axis starts at zero
        ax.set_ylim(0, 1.1)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x*100)}%"))

        # Rename the x-axis labels
        ax.set_xticklabels(["No", "Yes"])

    plt.tight_layout()

    plt.savefig(
        "../results/figures2/success_rate_by_vertical_conflict_bar.pdf",
        bbox_inches="tight",
    )


# Call the function with your data
plot_success_rate_by_conflict_with_dh_bar(success_rate_group_dh)




def plot_heatmap_success_rate(success_rate_group_ct):
    # Prepare the data
    heatmap_data = success_rate_group_ct.pivot_table(
        values='success_rate',
        index=['model_name', 'agent_type', 'experience_library'],
        columns='conflict_type',
        aggfunc='mean'
    )
    
    # Create a custom index label
    heatmap_data.index = [f"{m} - {a.replace('_', ' ').title()}{' + Exp' if e else ''}" 
                          for m, a, e in heatmap_data.index]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                linewidths=0.5, cbar_kws={'label': 'Success Rate'})
    
    plt.title('Success Rate by Model Configuration and Conflict Type')
    plt.xlabel('Conflict Type')
    plt.ylabel('Model Configuration')
    
    plt.tight_layout()
    plt.savefig('../results/figures2/heatmap_success_rate_by_conflict_type.pdf', bbox_inches='tight')
    plt.close()

# Assuming your data is in a DataFrame called 'success_rate_data'
plot_heatmap_success_rate(success_rate_group_ct)
