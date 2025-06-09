import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shap
from datetime import datetime
import numpy as np

csv_path = sys.argv[1]


data = pd.read_csv(csv_path)
# Rename columns
data.rename(columns={
        'Sleep Time Score_x': 'Sleep Time Score',
        'Performance_x': 'Performance',
        'Date_x': 'Date'
    }, inplace=True)

# Convert the 'Date' column to datetime format (they're already in ISO format YYYY-MM-DD)
data['Date'] = pd.to_datetime(data['Date'])
# Check for any dates that failed to parse
if data['Date'].isna().any():
    print("Warning: Some dates couldn't be parsed. Sample problematic dates:")
    print(data[data['Date'].isna()]['Date'].head())
    data = data.dropna(subset=['Date'])


# Derive output dir: static/plots/<filename_without_ext>
base_name = os.path.splitext(os.path.basename(csv_path))[0]
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))
os.makedirs(output_dir, exist_ok=True)


# Use consistent colors throughout the report
cluster_colors = {
    0: '#1a2a6c',  # Dark blue
    1: '#b21f1f',  # Magenta/red
    2: '#1e988a'   # Teal
}



def get_dates_up_to_cutoff(data, cutoff_date=None):
    """
    Get all unique dates from the dataset up to the specified cutoff date.
    If no cutoff date is provided, uses the last date in the dataset.

    Parameters:
    data (DataFrame): The dataset to filter
    cutoff_date (str): Optional cutoff date in 'YYYY-MM-DD' format

    Returns:
    list: List of datetime objects representing unique dates up to the cutoff
    """
    if cutoff_date is None:
        cutoff = data['Date'].max()
    else:
        cutoff = pd.to_datetime(cutoff_date)
    filtered_data = data[data['Date'] <= cutoff]
    return sorted(filtered_data['Date'].unique())

def get_match_loss_dates(data, cutoff_date=None):
    """
    Get all unique dates where Match_outcome is 2 (losses) up to cutoff date.
    If no cutoff date is provided, uses the last date in the dataset.

    Parameters:
    data (DataFrame): The dataset to filter
    cutoff_date (str): Optional cutoff date in 'YYYY-MM-DD' format

    Returns:
    list: List of datetime objects representing match loss dates
    """
    if cutoff_date is None:
        cutoff = data['Date'].max()
    else:
        cutoff = pd.to_datetime(cutoff_date)
    loss_dates = data[(data['Match_Outcome'] == 2) & (data['Date'] <= cutoff)]['Date'].unique()
    return sorted(loss_dates)

### work on this with HTML page with drop down menu where we must call all the dates from CSV file
 # User-provided cutoff date (optional - set to None to use all dates)
cutoff_date = None  # Example: "2025-04-16" or None for all dates

# Get all dates up to cutoff and match loss dates
dates_of_interest = get_dates_up_to_cutoff(data, cutoff_date)
loss_dates = get_match_loss_dates(data, cutoff_date)
print(f"Using dates from {dates_of_interest[0].date()} to {dates_of_interest[-1].date()}")
#if loss_dates:
#    print(f"Match loss dates: {[d.date() for d in loss_dates]}")
#else:
#    print("No match losses found in the selected date range")


# Define the dates of interest in ISO format
#dates_of_interest = [
#    "2025-01-14", "2025-01-22", "2025-01-29",
#    "2025-02-05", "2025-02-26", "2025-03-05",
#    "2025-03-12", "2025-03-19", "2025-03-26", "2025-04-02", "2025-04-09","2025-04-16"
#]
#dates_of_interest = [pd.to_datetime(date) for date in dates_of_interest]


# Initialize dictionaries to store the aggregate scores
aggregate_scores_median = {
    "Date": [],
    "Median_Self_Reported_Score": [],
    "Median_Biomarker_Score": [],
    "Median_Total_Score": []
}

aggregate_scores_mean = {
    "Date": [],
    "Mean_Self_Reported_Score": [],
    "Mean_Biomarker_Score": [],
    "Mean_Total_Score": []
}

# Define exclusions (in ISO format)
exclusions = {
    4: ["2025-02-26", "2025-03-05", "2025-03-12","2025-03-19","2025-03-26", "2025-04-02", "2025-04-09" ],
    2: ["2025-02-26", "2025-03-05", "2025-03-12"],
    11: ["2025-03-19","2025-03-26", "2025-04-02", "2025-04-09"],
    1: ["2025-03-26", "2025-04-09","2025-04-16"]  # Added exclusions for Player 1
}

# Convert exclusion dates to datetime
for player in exclusions:
    exclusions[player] = [pd.to_datetime(date) for date in exclusions[player]]

# Calculate aggregate scores for each date
for date in dates_of_interest:
    filtered_df = data[data['Date'] == date]

    # Apply exclusions
    for player, dates in exclusions.items():
        if date in dates:
            filtered_df = filtered_df[filtered_df['Player'] != player]

    # Calculate median values
    median_self_reported = filtered_df['Self_Reported_Score'].median()
    median_biomarker = filtered_df['Biomarker_Score'].median()
    median_total = filtered_df['Total_Score'].median()

    # Append to the results
    aggregate_scores_median['Date'].append(date)
    aggregate_scores_median['Median_Self_Reported_Score'].append(median_self_reported)
    aggregate_scores_median['Median_Biomarker_Score'].append(median_biomarker)
    aggregate_scores_median['Median_Total_Score'].append(median_total)

# Convert the results to DataFrames
aggregate_df = pd.DataFrame(aggregate_scores_median)

# Configure plot styling
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 300
# 1. Aggregate Timeline Visualization
def create_aggregate_timeline():
    # Create plot with custom colors
    plt.figure(figsize=(12, 6))  # Increased from 11, 5.5
    plt.plot(aggregate_df['Date'], aggregate_df['Median_Self_Reported_Score'],
             marker='o', color='#b21f1f', label='Self Reported Score', markersize=10, linewidth=3)
    plt.plot(aggregate_df['Date'], aggregate_df['Median_Biomarker_Score'],
             marker='o', color='#1a2a6c', label='Biomarker Score', markersize=10, linewidth=3)
    plt.plot(aggregate_df['Date'], aggregate_df['Median_Total_Score'],
             marker='o', color='#1e988a', label='Total Score', markersize=10, linewidth=3)

    # Mark match losses
    loss_dates = [pd.to_datetime(date, format='%d/%m/%Y') for date in ["22/01/2025", "5/02/2025", "12/03/2025","09/04/2025"]]
    for date in loss_dates:
        plt.axvline(x=date, color='red', linestyle='dotted', linewidth=2, alpha=0.7, label='Match Lost' if date == loss_dates[0] else "")

    plt.xlabel('Date', fontsize=22)
    plt.ylabel('Median Score', fontsize=22)
    plt.title('Aggregate Scores Timeline', fontsize=24, fontweight='bold')
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)

    # Add background grid for readability
    plt.grid(True, alpha=0.2, linestyle='--')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=20, framealpha=0.9)
    plt.tight_layout()
    #here we need to save in particular folder
    output_file = os.path.join(output_dir, 'plotaggregate.png')
    plt.savefig(output_file)
    return 0


# 2. Classification Visualization
def create_classification_plot():
    filtered_data = data[data['Day'].isin(['Post_game', 'Recovery'])]

    features = [
        'Sleep quality', 'Sleep Time Score', 'Tiredness', 'Energy level',
        'Stress level', 'Calmness', 'Anger', 'Muscle Soreness', 'Salivary pH',
        'Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Cortisol ng/ml',
        'Testosterone pg/ml', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score',
        'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score', 'Total_Score', 'Performance'
    ]

    pca_data = filtered_data[features].dropna()
    player_ids = filtered_data.loc[pca_data.index, 'Player']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(principal_components)

    pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    pca_df['Player'] = player_ids.values

    plt.figure(figsize=(12, 6))  # Increased from 11, 5.5
    palette = [cluster_colors[0], cluster_colors[1], cluster_colors[2]]
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'].map({0: 0, 1: 1, 2: 2}), cmap=plt.cm.colors.ListedColormap(palette), s=130, alpha=0.8)  # Increased size

    # Add cluster labels
    for cluster_id in range(3):
        cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
        centroid = cluster_data[['PC1', 'PC2']].mean()
        plt.annotate(f"Cluster {cluster_id}",
                     xy=(centroid['PC1'], centroid['PC2']),
                     xytext=(centroid['PC1'], centroid['PC2']),
                     fontsize=22,  # Increased from 20
                     color='white',
                     weight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc=palette[cluster_id], ec="none", alpha=0.7))

    # Add player labels
    for i, row in pca_df.iterrows():
        plt.annotate(f"{int(row['Player'])}",
                     xy=(row['PC1'], row['PC2']),
                     xytext=(row['PC1'], row['PC2']),
                     fontsize=20,  # Increased from 18
                     color='white',
                     weight='bold',
                     ha='center', va='center')

    plt.title('Player Classification', fontsize=24, fontweight='bold')  # Increased from 22
    plt.xlabel('Principal Component 1', fontsize=22)  # Increased from 20
    plt.ylabel('Principal Component 2', fontsize=22)  # Increased from 20
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    #here we need to save in particular folder
    output_file = os.path.join(output_dir, 'plotPC.png')
    plt.savefig(output_file)
    plt.close()
    return 0

def create_shap_analysis():
    filtered_data = data[data['Day'].isin(['Post_game', 'Recovery'])]

    features = [
        'Sleep quality', 'Sleep Time Score', 'Tiredness', 'Energy level',
        'Stress level', 'Calmness', 'Anger', 'Muscle Soreness', 'Salivary pH',
        'Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Cortisol ng/ml',
        'Testosterone pg/ml'
    ]

    target = 'Performance'

    model_data = filtered_data[features + [target]].dropna()
    X = model_data[features]
    y = model_data[target]

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42, max_depth=3)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 6))  # Increased from 11, 5.5

    # Custom SHAP summary plot with our color scheme
    feature_names = X.columns
    feature_importance = np.abs(shap_values).mean(0)
    sorted_idx = np.argsort(feature_importance)

    bars = plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='#1a2a6c', alpha=0.8, height=0.6)
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], fontsize=22)  # Increased from 20
    plt.xlabel('mean(|SHAP value|)', fontsize=22)  # Increased from 20
    plt.title('Feature Importance (SHAP Values)', fontsize=24, fontweight='bold')  # Increased from 22
    plt.grid(True, axis='x', alpha=0.2, linestyle='--')
    plt.xticks(fontsize=20)  # Increased from 18

    # Add value labels to the bars with improved formatting
    for i, v in enumerate(feature_importance[sorted_idx]):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=20, fontweight='bold')  # Increased from 18

    plt.tight_layout()
    #here we need to save in particular folder
    output_file = os.path.join(output_dir, 'plotSHAP.png')
    plt.savefig(output_file)
    plt.close()
    return 0


# Generate all components
create_aggregate_timeline()
create_classification_plot()
create_shap_analysis()




