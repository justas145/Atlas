import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.font_manager as font_manager


def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def setup_matplotlib():
    # Set default font and plot styles
    matplotlib.rc("font", size=12, family="Ubuntu")
    matplotlib.rc("lines", linewidth=2, markersize=8)
    matplotlib.rc("grid", color="darkgray", linestyle=":")


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
        if 'gpt-4o' in model_name:
            return 'GPT-4o'
        elif 'llama' in model_name:
            return 'Llama-70B'
        return model_name

    # Apply the custom function to the model_name column
    df["model_name"] = df["model_name"].apply(replace_model_name)
    

    
    return df


def calculate_scores(df):

    # Total score by model, agent type, experience library, and num_aircraft
    total_scores = (
        df.groupby(["model_name", "agent_type", "experience_library", "num_aircraft", "conflict_type"])
        .score.sum()
        .reset_index()
    )

    # Average score by model, agent type, experience library, and num_aircraft
    average_scores = (
        df.groupby(["model_name", "agent_type", "experience_library", "num_aircraft", "conflict_type"])
        .score.mean()
        .reset_index()
    )

    average_num_tools_used = (
        df.groupby(["model_name","agent_type","experience_library","num_aircraft","conflict_type"])
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
    ).fillna(
        0
    )  # Fill NaN values with 0 for success_count
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
    ).fillna(
        0
    )  # Fill NaN values with 0 for success_count

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
    ).fillna(
        0
    )  # Fill NaN values with 0 for success_count

    success_rate_group_ct["success_rate"] = (
        success_rate_group_ct["success_count"] / success_rate_group_ct["total_count"]
    )

    return (
        total_scores,
        average_scores,
        success_rate,
        success_rate_group_ac,
        success_rate_group_ct,
        average_num_tools_used,
        average_num_commands_sent,
    )


def plot_total_scores(df):
    plt.figure(figsize=(10, 6))
    # Create a formatted label for better legend readability
    df["hue_label"] = (
        df["agent_type"].str.replace("_", " ").str.title()
        + " with"
        + df["experience_library"].apply(
            lambda x: " Experience" if x else "out Experience"
        )
    )
    sns.barplot(
        x="model_name",
        y="score",
        hue="hue_label",
        data=df,
        palette="husl",
        errorbar=None,
        estimator=sum,
    )
    plt.title("Total Score by Model, Agent Type, and Experience Library")
    plt.xlabel("Model Name")
    plt.ylabel("Total Score")
    plt.legend(framealpha=0.15,title="Agent Type/Experience")
    # plt.xticks(rotation=45)
    # Automatically adjust layout to avoid cutting off elements
    
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(figures_directory, "bar_chart_total_score.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_average_scores(df):
    plt.figure(figsize=(10, 6))
    # Create a new column for hue that combines and formats the text
    df["hue_label"] = (
        df["agent_type"].str.replace("_", " ").str.title()
        + " with"
        + df["experience_library"].apply(
            lambda x: " Experience" if x else "out Experience"
        )
    )
    sns.barplot(
        x="model_name",
        y="score",
        hue="hue_label",
        data=df,
        palette="husl",
        errorbar=None,
    )
    plt.title("Average Score by Model")
    plt.xlabel("Model Name")
    plt.ylabel("Average Score")
    plt.legend(framealpha=0.15,title="Agent Type/Experience")
    # plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Automatically adjust layout to avoid cutting off elements

    save_path = os.path.join(figures_directory, "bar_chart_avg_score.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_average_num_tools_used(df):
    plt.figure(figsize=(10, 6))
    # Create a new column for hue that combines and formats the text
    df["hue_label"] = (
        df["agent_type"].str.replace("_", " ").str.title()
        + " with"
        + df["experience_library"].apply(
            lambda x: " Experience" if x else "out Experience"
        )
    )
    sns.barplot(
        x="model_name",
        y="num_total_commands",
        hue="hue_label",
        data=df,
        palette="husl",
        errorbar=None,
    )
    plt.title("Average Number of Tools Used by Model")
    plt.xlabel("Model Name")
    plt.ylabel("Average Number of Tools Used")
    plt.legend(framealpha=0.15,title="Agent Type/Experience")
    # plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Automatically adjust layout to avoid cutting off elements
    
    save_path = os.path.join(figures_directory, "bar_chart_avg_num_tools_used.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_average_num_commands_sents(df):
    plt.figure(figsize=(10, 6))
    # Create a new column for hue that combines and formats the text
    df["hue_label"] = (
        df["agent_type"].str.replace("_", " ").str.title()
        + " with"
        + df["experience_library"].apply(
            lambda x: " Experience" if x else "out Experience"
        )
    )
    sns.barplot(
        x="model_name",
        y="num_send_commands",
        hue="hue_label",
        data=df,
        palette="husl",
        errorbar=None,
    )
    plt.title("Average Number of Commands Sent by Model")
    plt.xlabel("Model Name")
    plt.ylabel("Average Number of Commands Sent")
    plt.legend(framealpha=0.15,title="Agent Type/Experience")
    # plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Automatically adjust layout to avoid cutting off elements
    
    save_path = os.path.join(figures_directory, "bar_chart_avg_num_commands_sent.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_success_rates(df):
    plt.figure(figsize=(10, 6))
    # Create a formatted label for better legend readability
    df["hue_label"] = (
        df["agent_type"].str.replace("_", " ").str.title()
        + " with"
        + df["experience_library"].apply(
            lambda x: " Experience" if x else "out Experience"
        )
    )
    sns.barplot(
        x="model_name",
        y="success_rate",
        hue="hue_label",
        data=df,
        palette="husl",
        errorbar=None,
    )
    plt.title("Success Rate by Model")
    plt.xlabel("Model Name")
    plt.ylabel("Success Rate")
    plt.legend(framealpha=0.15,title="Agent Type/Experience")
    # plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Automatically adjust layout to avoid cutting off elements
    
    save_path = os.path.join(figures_directory, "bar_chart_success_rate.pdf")
    plt.savefig(save_path)
    plt.close()


def plot_total_score_by_aircraft(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[
            df["model_name"] == model
        ].copy()  # Make a copy to avoid SettingWithCopyWarning
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.lineplot(
            x="num_aircraft",
            y="score",
            hue="hue_label",
            data=model_data,
            marker="o",
            palette="husl",
            errorbar=None,
            estimator=sum,
        )
        plt.title(f"Total Score by Number of Aircraft for {model}")
        plt.xlabel("Number of Aircraft")
        plt.ylabel("Total Score")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(
            figures_directory,
            f"total_score_by_aircraft_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_average_score_by_aircraft(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.lineplot(
            x="num_aircraft",
            y="score",
            hue="hue_label",
            data=model_data,
            marker="o",
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Score by Number of Aircraft for {model}")
        plt.xlabel("Number of Aircraft")
        plt.ylabel("Average Score")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(
            figures_directory,
            f"average_score_by_aircraft_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_average_num_tools_used_by_aircraft(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.lineplot(
            x="num_aircraft",
            y="num_total_commands",
            hue="hue_label",
            data=model_data,
            marker="o",
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Number of Tools Used by Number of Aircraft for {model}")
        plt.xlabel("Number of Aircraft")
        plt.ylabel("Average Number of Tools Used")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(
            figures_directory,
            f"average_num_tools_used_by_aircraft_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_average_num_commands_sent_by_aircraft(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.lineplot(
            x="num_aircraft",
            y="num_send_commands",
            hue="hue_label",
            data=model_data,
            marker="o",
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Number of Commands Sent by Number of Aircraft for {model}")
        plt.xlabel("Number of Aircraft")
        plt.ylabel("Average Number of Commands Sent")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(
            figures_directory,
            f"average_num_commands_sent_by_aircraft_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_success_rate_by_aircraft(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.lineplot(
            x="num_aircraft",
            y="success_rate",
            hue="hue_label",
            data=model_data,
            marker="o",
            palette="husl",
            errorbar=None,
        )

        plt.title(f"Success Rate by Number of Aircraft for {model}")
        plt.xlabel("Number of Aircraft")
        plt.ylabel("Success Rate")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        plt.grid(True)
        plt.tight_layout()

        # Ensure that the y-axis starts at zero
        plt.ylim(0, 1)

        save_path = os.path.join(
            figures_directory,
            f"success_rate_by_aircraft_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_total_score_by_conflict_type(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.barplot(
            x="conflict_type",
            y="score",
            hue="hue_label",
            data=model_data,
            palette="husl",
            errorbar=None,
            estimator=sum,
        )
        plt.title(f"Total Score by Conflict Type for {model}")
        plt.xlabel("Conflict Type")
        plt.ylabel("Total Score")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # Automatically adjust layout to avoid cutting off elements
        
        save_path = os.path.join(
            figures_directory,
            f"total_score_by_conflict_type_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_average_score_by_conflict_type(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.barplot(
            x="conflict_type",
            y="score",
            hue="hue_label",
            data=model_data,
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Score by Conflict Type for {model}")
        plt.xlabel("Conflict Type")
        plt.ylabel("Average Score")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # Automatically adjust layout to avoid cutting off elements
        
        save_path = os.path.join(
            figures_directory,
            f"average_score_by_conflict_type_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()

def plot_average_num_tools_sent_by_conflict_type(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.barplot(
            x="conflict_type",
            y="num_total_commands",
            hue="hue_label",
            data=model_data,
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Number of Tools Used by Conflict Type for {model}")
        plt.xlabel("Conflict Type")
        plt.ylabel("Average Number of Tools Used")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # Automatically adjust layout to avoid cutting off elements
        
        save_path = os.path.join(
            figures_directory,
            f"average_num_tools_used_by_conflict_type_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()

def plot_average_num_commands_sent_by_conflict_type(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.barplot(
            x="conflict_type",
            y="num_send_commands",
            hue="hue_label",
            data=model_data,
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Average Number of Commands Sent by Conflict Type for {model}")
        plt.xlabel("Conflict Type")
        plt.ylabel("Average Number of Commands Sent")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # Automatically adjust layout to avoid cutting off elements
        
        save_path = os.path.join(
            figures_directory,
            f"average_num_commands_sent_by_conflict_type_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


def plot_success_rate_by_conflict_type(df):
    models = df["model_name"].unique()
    for model in models:
        plt.figure(figsize=(12, 8))
        model_data = df[df["model_name"] == model].copy()
        model_data.loc[:, "hue_label"] = (
            model_data["agent_type"].str.replace("_", " ").str.title()
            + " with"
            + model_data["experience_library"].apply(
                lambda x: " Experience" if x else "out Experience"
            )
        )
        sns.barplot(
            x="conflict_type",
            y="success_rate",
            hue="hue_label",
            data=model_data,
            palette="husl",
            errorbar=None,
        )
        plt.title(f"Success Rate by Conflict Type for {model}")
        plt.xlabel("Conflict Type")
        plt.ylabel("Success Rate")
        plt.legend(framealpha=0.15,title="Agent Type/Experience")
        # plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        # Automatically adjust layout to avoid cutting off elements
        
        save_path = os.path.join(
            figures_directory,
            f"success_rate_by_conflict_type_{model.replace(':', '_').replace(' ', '_')}.pdf",
        )
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    setup_matplotlib()
    csv_path = "../results/main/final_combined_V2.csv"
    figures_directory = "../results/figures"
    df = load_and_prepare_data(csv_path)
    total_scores, average_scores, success_rate, success_rate_group_ac, success_rate_group_ct, average_num_tools_used, average_num_commands_sent = (
        calculate_scores(df)
    )

    plot_total_scores(total_scores) 
    plot_average_scores(average_scores) 
    plot_average_num_tools_used(average_num_tools_used)
    plot_average_num_commands_sents(average_num_commands_sent)
    plot_success_rates(success_rate) 

    plot_total_score_by_aircraft(total_scores) 
    plot_average_score_by_aircraft(average_scores) 
    plot_average_num_tools_used_by_aircraft(average_num_tools_used)
    plot_average_num_commands_sent_by_aircraft(average_num_commands_sent)
    plot_success_rate_by_aircraft(success_rate_group_ac) 

    plot_total_score_by_conflict_type(total_scores) 
    plot_average_score_by_conflict_type(average_scores) 
    plot_average_num_tools_sent_by_conflict_type(average_num_tools_used)
    plot_average_num_commands_sent_by_conflict_type(average_num_commands_sent)
    plot_success_rate_by_conflict_type(success_rate_group_ct) 
