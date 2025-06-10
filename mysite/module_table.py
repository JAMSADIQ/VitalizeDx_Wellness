import pandas as pd
import matplotlib.pyplot as plt

##############some function for aggregate PCA and shap plots
def get_dates_up_to_cutoff(data, cutoff_date=None):
    """
    Get all unique dates from the dataset up to the specified cutoff date.
    If no cutoff date is provided, uses the last date in the dataset.

    Parameters:
    data (DataFrame): The dataset (with 'Date' column in datetime format)
    cutoff_date (str): Optional cutoff date in 'm/d/Y' format (e.g., '1/15/2025')

    Returns:
    list: List of datetime objects representing unique dates up to the cutoff
    """
    if cutoff_date is None:
        cutoff = data['Date'].max()
    else:
        # Force parsing as m/d/Y (month first)
        cutoff = pd.to_datetime(cutoff_date, format='%m/%d/%Y')

    filtered_data = data[data['Date'] <= cutoff]
    #filtered_data = data[(data['Date'] <= cutoff) & (data['Day'] == 'Recovery')]
    return sorted(filtered_data['Date'].unique())

def get_match_loss_dates(data, cutoff_date=None):
    """
    Get all unique dates where Match_outcome is 2 (losses) up to cutoff date.
    If no cutoff date is provided, uses the last date in the dataset.

    Parameters:
    data (DataFrame): The dataset (with 'Date' column in datetime format)
    cutoff_date (str): Optional cutoff date in 'm/d/Y' format (e.g., '1/15/2025')

    Returns:
    list: List of datetime objects representing match loss dates
    """
    if cutoff_date is None:
        cutoff = data['Date'].max()
    else:
        # Force parsing as m/d/Y (month first)
        cutoff = pd.to_datetime(cutoff_date, format='%m/%d/%Y')
    loss_dates = data[(data['Match_Outcome'] == 2) &
                     (data['Day'] == 'Recovery') &
                     (data['Date'] <= cutoff)]['Date'].unique()
    return sorted(loss_dates)


#clasification table for classification template
def create_classification_table(pca_df, data, cluster_colors):
    # Get most common cluster for each player
    player_clusters = pca_df.groupby('Player')['Cluster'].agg(lambda x: x.mode()[0]).reset_index()

    # Count players in each cluster
    cluster_counts = player_clusters['Cluster'].value_counts().sort_index()

    # Create cluster summary table with our color scheme
    summary_table = f"""
    <div style="margin-bottom: 15px;">
        <table style="width:100%; font-size: 9pt; border-collapse: collapse; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <tr>
                <th style="width:33%; background-color: #1a2a6c; color: white; padding: 8px; text-align: center;">Cluster 0 (MED)</th>
                <th style="width:33%; background-color: #b21f1f; color: white; padding: 8px; text-align: center;">Cluster 1 (TOP)</th>
                <th style="width:33%; background-color: #1e988a; color: white; padding: 8px; text-align: center;">Cluster 2 (LOW)</th>
            </tr>
            <tr>
                <td style="text-align: center; padding: 8px; background-color: rgba(26, 42, 108, 0.1);">{cluster_counts.get(0, 0)} players</td>
                <td style="text-align: center; padding: 8px; background-color: rgba(178, 31, 31, 0.1);">{cluster_counts.get(1, 0)} players</td>
                <td style="text-align: center; padding: 8px; background-color: rgba(30, 152, 138, 0.1);">{cluster_counts.get(2, 0)} players</td>
            </tr>
        </table>
    </div>
    """

    # Get cluster characteristics
    features = [
        'Sleep quality', 'Sleep Time Score', 'Tiredness', 'Energy level',
        'Stress level', 'Calmness', 'Anger', 'Muscle Soreness', 'Performance',
        'Self_Reported_Score', 'Biomarker_Score', 'Total_Score'
    ]

    cluster_data = data[data['Player'].between(1, 13)].copy()
    cluster_data = cluster_data.merge(player_clusters, on='Player')

    cluster_stats = cluster_data.groupby('Cluster')[features].mean().round(2)

    # Create an enhanced HTML table with our styling
    table_html = """
    <table style="width:100%; font-size: 9pt; border-collapse: collapse; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <tr style="background: linear-gradient(135deg, #1a2a6c, #b21f1f, #1e988a); color: white;">
            <th style="padding: 8px; text-align: center; width:15%;">Player</th>
            <th style="padding: 8px; text-align: center; width:15%;">Cluster</th>
            <th style="padding: 8px; text-align: center; width:70%;">Key Characteristics</th>
        </tr>
    """

    for player in range(1, 14):
        if player in player_clusters['Player'].values:
            cluster = player_clusters[player_clusters['Player'] == player]['Cluster'].values[0]
            char = cluster_stats.loc[cluster].to_dict()

            # Format key metrics with highlighting
            highlighted_metrics = []
            for k, v in char.items():
                color = ""
                if k in ['Total_Score', 'Biomarker_Score', 'Self_Reported_Score', 'Performance'] and v > 75:
                    color = "color: #4caf50; font-weight: bold;"
                elif k in ['Total_Score', 'Biomarker_Score', 'Self_Reported_Score', 'Performance'] and v < 50:
                    color = "color: #f44336; font-weight: bold;"

                highlighted_metrics.append(f"<span style='{color}'>{k}: {v}</span>")

            char_str = "<br>".join(highlighted_metrics)

            # Get cell background color based on cluster
            cluster_bg_color = f"rgba({', '.join(map(str, [int(x*255) for x in plt.cm.colors.to_rgb(cluster_colors[cluster])]))}, 0.2)"
            # Get cluster color
            cluster_color = cluster_colors[cluster]

            table_html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px; text-align: center; font-weight: bold;">{player}</td>
                <td style="background-color: {cluster_color};
                    color: white; text-align: center; font-weight: bold; padding: 8px;">{cluster}</td>
                <td style="padding: 8px; font-size: 5pt; background-color: {cluster_bg_color};">{char_str}</td>
            </tr>
            """
        else:
            table_html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px; text-align: center; font-weight: bold;">{player}</td>
                <td style="text-align: center; padding: 8px;">N/A</td>
                <td style="padding: 8px; font-size: 5pt;">No data available</td>
            </tr>
            """

    table_html += "</table>"
    return summary_table + table_html