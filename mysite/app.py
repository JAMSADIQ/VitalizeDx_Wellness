import os
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import timedelta
from datetime import datetime
import subprocess
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import csv
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO
import sys
from scipy.stats import norm
from module_table import create_classification_table, get_dates_up_to_cutoff, get_match_loss_dates


########### set the path  where we  have app.py  script###################
path = '/home/vitalizedx/mysite'
if path not in sys.path:
    sys.path.append(path)

app = Flask(__name__)
app.secret_key = 'd2c8f1e7b5a9d4f6c3e8b2a7d5f1e3c'  # Change this to a strong random key in production

# Session security
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# Configuration
UPLOAD_FOLDER = '/home/vitalizedx/mysite/uploads'
PLOT_FOLDER = '/home/vitalizedx/mysite/static'
HTML_FOLDER = '/home/vitalizedx/mysite/templates'
DATA_FILE_FOLDER  = '/home/vitalizedx/mysite/data'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['HTML_FOLDER'] = HTML_FOLDER
app.config['DATA_FILE_FOLDER'] = DATA_FILE_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#if uplaod files the can have these extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}


# User configuration (replace with database in production)
users = {
    'admin': {
        'password': 'mypass',  # Make sure this is exactly what you're typing
        'name': 'admin'
    },
    'user': {
       'password': 'userpass',  # Make sure this is exactly what you're typing
       'name': 'user'
   },
    'RiminiWellness25': {
        'password': 'BeVitalizeDx!',  # Make sure this is exactly what you're typing
        'name': 'RiminiWellness25'
    },
    'Summer25': {
        'password': 'BeVitalizeDx!',  # Make sure this is exactly what you're typing
        'name': 'Summer25'
    }
}


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    pass



##############some function for aggregate PCA and shap plots
def change_get_dates_up_to_cutoff(data, cutoff_date=None):
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

def change_get_match_loss_dates(data, cutoff_date=None):
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

############################################################
#clasification table
def change_create_classification_table(pca_df, data, cluster_colors):
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

###################################################################
### New stuff for Score Card for player
# Define colors
RED = '#E63946'
DARK_BLUE = '#1D3557'
GOLD = '#FFD166'


#############################################################
@login_manager.user_loader
def user_loader(username):
    if username not in users:
        return
    user = User()
    user.id = username
    user.name = users[username]['name']
    return user

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_existing_files():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        return [f for f in os.listdir(app.config['UPLOAD_FOLDER'])
               if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
    return []


@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/test-auth')
def test_auth():
    test_user = 'admin'
    test_pass = 'yourpassword'
    if test_user in users and users[test_user]['password'] == test_pass:
        return "Credentials are correctly set in code!"
    return "Credentials DON'T MATCH in code!"

############new code
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(f"Attempted login - Username: '{username}', Password: '{password}'")  # Debug line
        print(f"Login attempt - Username: {username}, Password: {password}")  # Debug
        print(f"Existing users: {users.keys()}")  # Debug
        if username in users and users[username]['password'] == password:
            print("Credentials matched!")  # Debug
            user = User()
            user.id = username
            login_user(user)
            flash(f'Welcome back, {users[username]["name"]}!', 'success')

            # Check if username is 'LaForge25' and redirect accordingly
            if username == 'LaForge25':  # Remove .lower() since you want exact match
                return redirect(url_for('player_dashboard'))  # Route for player page
            else:
                return redirect(url_for('dashboard'))  # Route for regular users

        flash('Invalid username or password', 'error')
    return render_template('login.html')

# Add this route for player dashboard
@app.route('/player')  # Changed from /coach to /player
@login_required
def player_dashboard():  # Changed function name to match
    return render_template('player_dashboard.html')

################3new code
@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get list of existing files
    existing_files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        existing_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])
                        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]

    return render_template('dashboard.html', existing_files=existing_files)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Store questionnaire responses (in production, use a database)
questionnaire_responses = {}

@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    if request.method == 'POST':
        try:
            # Store the responses
            filename = request.form.get('filename')
            weight = request.form.get('weight')
            height = request.form.get('height')
            bmi_result = request.form.get('bmi')
            responses = {
            'q1': int(request.form.get('q1')),
            'q2': int(request.form.get('q2')),
            'q3': int(request.form.get('q3')),
            'q4': int(request.form.get('q4')),
            'q5': int(request.form.get('q5')),
            'q6': int(request.form.get('q6')),
            'q7': int(request.form.get('q7')),
            'comments': request.form.get('comments', ''),
            # BMI fields with proper error handling
             'gender': request.form.get('gender', ''),
                'weight': float(weight) if weight else None,
                'weight_unit': request.form.get('weight_unit', 'kg'),
                'height': float(height) if height else None,
                'height_unit': request.form.get('height_unit', 'cm'),
                'bmi': float(bmi_result) if bmi_result else None
            }

            questionnaire_responses[filename] = responses
            flash('Questionnaire submitted successfully!', 'success')
            return redirect(url_for('show_sequence', page_index=0))
        except (ValueError, TypeError) as e:
            flash('Error processing form data. Please check your inputs.', 'error')
            return redirect(url_for('questionnaire', filename=request.form.get('filename')))

    filename = request.args.get('filename')
    return render_template('questionnaire.html', filename=filename)


#########for score card ########################
def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def gaussianplot(self_report):
    x = np.linspace(0, 100, 1000)

    # Define the Gaussian parameters
    distributions = [
        #{"title": "Biomarker", "mean": 85, "sigma": 15, "color": "blue", "ref_value": 82, "dash": "solid"},
        {"title": "Self Reported", "mean": 50, "sigma": 25, "color": "green", "ref_value": self_report , "dash": "dot"}
        #,{"title": "Total Score", "mean": 75, "sigma": 20, "color": "red", "ref_value": 72, "dash": "dash"},
    ]

    # Create the figure
    fig = go.Figure()

    # Add each Gaussian to the plot
    for dist in distributions:
        y = norm.pdf(x, dist["mean"], dist["sigma"])

        # Plot the Gaussian curve with a unique line style
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=dist["title"],
            line=dict(color=dist["color"], width=2, dash=dist["dash"]),
            hoverinfo='name+x+y'
        ))

        # Add subtle vertical line for the reference value
        fig.add_vline(
            x=dist["ref_value"],
            line=dict(color=dist["color"], width=2, dash="dash"),
            # Remove annotation to declutter
            annotation=None
        )

    # Final layout tweaks
    fig.update_layout(
        title="Report {0}".format(self_report),
        xaxis_title="Score",
        yaxis_title="Probability Density",
        xaxis_range=[min(0, self_report), 105],
        showlegend=True,
        legend=dict(x=0.1, y=1),
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        gridcolor='lightgrey',
        showline=True,
        linecolor='black',
        linewidth=1
    ),
    yaxis=dict(
        gridcolor='lightgrey',
        showline=True,
        linecolor='black',
        linewidth=1
    )
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn',
    config={'staticPlot': True})


def prepare_radar_data(file_path):
    # Load the final output file
    data = pd.read_csv(file_path)

    # Rename columns for consistency
    data.rename(columns={
        'Sleep Time Score_x': 'Sleep Time Score',
        'Performance_x': 'Performance',
        'Date_x': 'Date'  # Ensure 'Date' is used instead of 'Date of Match'
    }, inplace=True)

    # Ensure the 'Date' column is properly parsed as a datetime format with MM/DD/YYYY
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y', errors='coerce')

        print(data['Date'].dt.strftime('%m/%d/%Y').dropna().unique())
    else:
        raise ValueError("The 'Date' column is missing from the dataset.")



    # Get reference data
    reference_row = data[data['Player'] == 1000].sort_values(by="Date", ascending=False).iloc[0].copy()

    # Compute metrics
    data['Sleep QxT'] = data['Sleep Time Score'] * data['Sleep quality']
    reference_row['Sleep QxT'] = reference_row['Sleep Time Score'] * reference_row['Sleep quality']

    # Columns for radar chart
    cols_to_normalize = ['Sleep QxT', 'Tiredness', 'Energy level', 'Stress level', 'Calmness', 'Anger', 'Muscle Soreness']

    # Normalize values (0-100 scale)
    for col in cols_to_normalize:
        data[col] = (data[col] / 10) * 100
        reference_row[col] = (reference_row[col] / 10) * 100

    return data, reference_row, cols_to_normalize

def load_and_prepare_timeline_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])

    # Special handling for Player 1 on 2025-04-09 - set biomarker scores to NaN
    mask = (data['Player'] == 1) & (data['Date'] == '2025-04-09')
    data.loc[mask, ['Biomarker_Score', 'Total_Score']] = np.nan

    return data.sort_values('Date')




def calculate_self_reported_score(row):
    """Calculate score for self-reported parameters"""
    scores = [
        row['Sleep quality'] / 10,  # Convert from 0-10 to 0-1 scale
        row['Sleep Time Score'],
        1 - (row['Tiredness'] / 10),  # Convert from 0-10 to 0-1 scale and invert
        row['Energy level'] / 10,
        1 - (row['Stress level'] / 10),
        row['Calmness'] / 10,
        1 - (row['Anger'] / 10),
        1 - (row['Muscle Soreness'] / 10)
    ]
    return np.mean(scores) * 100  # Convert to percentage

def calculate_salivary_ph_score(ph):
    """Calculate score for salivary pH"""
    if pd.isna(ph):  # Handle missing values
        return np.nan
    if ph == 0:  # New rule: if pH is 0, return 0
        return 0
    if 7 <= ph <= 7.5:
        return 100
    return 50

def calculate_nitrates_score(nitrates):
    """Calculate score for salivary nitrates"""
    if pd.isna(nitrates):  # Handle missing values
        return np.nan
    if nitrates == 0:  # New rule: if nitrates is 0, return 0
        return 0
    scores = {1: 50, 2: 75, 3: 100}
    return scores.get(nitrates, 0)

def calculate_uric_acid_score(uric_acid):
    """Calculate score for uric acid"""
    if pd.isna(uric_acid):  # Handle missing values
        return np.nan
    if uric_acid == 0:  # New rule: if uric acid is 0, return 0
        return 0
    if uric_acid < 0.5:
        return 1
    if uric_acid < 2.5:
        return 50
    elif 2.5 <= uric_acid <= 5:
        return 100
    else:
        return 50

def calculate_tcr_score(tcr):
    """Calculate score for TCR"""
    if pd.isna(tcr):  # Handle missing values
        return np.nan
    if tcr == 0:  # New rule: if TCR is 0, return 0
        return 0
    if tcr < 0.03:
        return 50
    elif 0.03 <= tcr <= 0.07:
        return 100
    else:
        return 120

def calculate_all_scores(df):
    """Calculate all scores for the dataset"""
    # Calculate self-reported scores
    df['Self_Reported_Score'] = df.apply(calculate_self_reported_score, axis=1)

    # Calculate individual biomarker scores (will return NaN for missing values)
    df['pH_Score'] = df['Salivary pH'].apply(calculate_salivary_ph_score)
    df['Nitrates_Score'] = df['Salivary Nitrates'].apply(calculate_nitrates_score)
    df['Uric_Acid_Score'] = df['Salivary Uric Acid mg/dl'].apply(calculate_uric_acid_score)
    df['TCR_Score'] = df['TCR'].apply(calculate_tcr_score)

    # Calculate average biomarker score (only for available values)
    df['Biomarker_Score'] = df[['pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score']].mean(axis=1, skipna=True)

    df['Biomarker_Factor'] = np.where(
        df[['pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score']].isna().all(axis=1),
        np.nan,  # If all biomarkers are missing
        np.where(
            (df['pH_Score'] == 0) | (df['Nitrates_Score'] == 0) |
            (df['Uric_Acid_Score'] == 0) | (df['TCR_Score'] == 0),
            0,  # If any biomarker is 0
            np.where(
                df['Biomarker_Score'] < 35,
                0,  # If score is below 35
                df['Biomarker_Score'].fillna(100) / 100  # Normal case
            )
        )
    )

    # Calculate total score only when we have biomarker data
    df['Total_Score'] = np.where(
        pd.isna(df['Biomarker_Factor']),
        np.nan,
        df['Self_Reported_Score'] * df['Biomarker_Factor'] * 1.2
    )

    return df



# Calculate deviations - FIXED FUNCTION
def calculate_deviation(current_data, averages):
    deviations = {}
    for player in current_data.index:
        player_deviations = {}
        for col in numeric_columns:
            try:
                # Handle potential Series objects by explicitly getting single values
                if isinstance(current_data.loc[player, col], pd.Series):
                    current_val = current_data.loc[player, col].iloc[0]
                else:
                    current_val = current_data.loc[player, col]

                if isinstance(averages.loc[player, col], pd.Series):
                    avg_val = averages.loc[player, col].iloc[0]
                else:
                    avg_val = averages.loc[player, col]

                # Now check if values are valid for calculation
                if pd.notna(current_val) and pd.notna(avg_val) and avg_val != 0:
                    player_deviations[col] = (current_val - avg_val) / avg_val
                else:
                    player_deviations[col] = np.nan
            except (KeyError, IndexError):
                player_deviations[col] = np.nan
        deviations[player] = player_deviations
    return pd.DataFrame.from_dict(deviations, orient='index')


def evaluate_player_data(player_data, deviation_data, date):
    # Benchmark data
    benchmark_data = {
    'Data': [
        'Sleep quality', 'How many hours did you sleep?', 'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level',
        'Calmness', 'Anger', 'Muscle Soreness', 'SalivaCoding', 'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl',
        'Traces Blood Saliva', 'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L', 'Testosterone pg/ml', 'Testosterone nmol/L',
        'TCR', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score',
        'Biomarker_Factor', 'Total_Score', 'AST U/L', 'CK U/L', 'DHEA-S ng/L', 'DHEA-S nmol/L'
    ],
    'Benchmark Mean': [
        8, None, 1, 1, 8, 1, 10, 1, 1, 2025012913, 7, 2, 5, 0.5, 1.5, 8, 22.08, 300, 1.041, 0.047, 82.5, 100, 100, 100, 100, 100, 1, 99, 30, 50, 5, 13.55
    ],
    'Benchmark Min': [
        7, None, 0.8, 0.8, 6, 0.8, 8, 0.8, 0.8, 2025012913, 6, 1, 2, 0, 0, 4, 14, 150, 0, 0, 64, 50, 50, 50, 50, 50, 0.5, 49.5, 15, 25, 2.5, 6.775
    ],
    'Benchmark Max': [
        9, None, 1.2, 1.2, 10, 1.2, 12, 1.2, 1.2, 2025012913, 8, 3, 7, 1, 3, 12, 30, 450, 2, 0.11, 100, 150, 150, 150, 150, 150, 1.5, 150, 58.5, 97.5, 9.75, 26.4225
    ]
    }

    benchmark_df = pd.DataFrame(benchmark_data)
    results = []
    all_players = player_data.index.unique()

    for player_name in all_players:
        row = player_data.loc[player_name] if player_name in player_data.index else pd.Series()

        for _, benchmark_row in benchmark_df.iterrows():
            parameter = benchmark_row['Data']
            if parameter not in player_data.columns:
                continue

            # Get player value (0 if missing for key biomarkers)
            key_biomarkers = ['Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Traces Blood Saliva',
                             'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L',
                             'Testosterone pg/ml', 'Testosterone nmol/L', 'TCR']

            if parameter in key_biomarkers:
                player_value = row.get(parameter, 0)  # Default to 0 for key biomarkers
                is_key_biomarker = True
            else:
                player_value = row.get(parameter, np.nan)  # NaN for others
                is_key_biomarker = False

            # Handle Series objects for player_value
            if isinstance(player_value, pd.Series):
                player_value = player_value.iloc[0] if not player_value.empty else np.nan

            # Skip if value is NaN and not a key biomarker
            if pd.isna(player_value) and not is_key_biomarker:
                continue

            # Get deviation if available
            if parameter in deviation_data.columns and player_name in deviation_data.index:
                deviation_val = deviation_data.loc[player_name, parameter]
                # Handle Series objects for deviation
                if isinstance(deviation_val, pd.Series):
                    deviation = deviation_val.iloc[0] if not deviation_val.empty else np.nan
                else:
                    deviation = deviation_val
            else:
                deviation = np.nan

            # Initialize associations and actions
            direct_assoc = []
            indirect_assoc = []
            actions = []
            # Parameter-specific checks (from your first version)
            if parameter == 'Sleep quality':
                if player_value < 6:
                    direct_assoc.append('Poor sleep quality')
                    indirect_assoc.append('Impacts tiredness, energy level, and stress')
                    actions.append('Improve sleep hygiene by avoiding digital media browsing during night time and minimize light sources in the bedroom')

            elif parameter == 'Sleep Time Score':
                if player_value < 1:
                    direct_assoc.append('Insufficient sleep duration')
                    indirect_assoc.append('Affects tiredness and muscle soreness')
                    actions.append('Aim for consistent 7-9 hours sleep')

            elif parameter == 'Tiredness':
                if player_value > 6:
                    direct_assoc.append('High tiredness reflects poor recovery')
                    indirect_assoc.append('Linked to poor sleep and impacts energy')
                    actions.append('Prioritize recovery increasing sleep time up to 10 hours, improve sleep by decreasing muscular soreness through cold water immersions 5 x 1 min at 11*C, or 15 min at 15*C')

            elif parameter == 'Energy level':
                if player_value < 6:
                    direct_assoc.append('Low energy')
                    indirect_assoc.append('Impacted by stress, sleep, and recovery')
                    actions.append('Improve diet by intaking 6g per day of L-alanyl-L-glutamine, 0.17g per kg BCAA (2:1:1) and 0.04g per kg arginine and manage stress through mindful meditation 10-20min per day')

            elif parameter == 'Stress level':
                if player_value > 5:
                    direct_assoc.append('High stress negatively impacts energy')
                    indirect_assoc.append('Increases muscle soreness and affects recovery')
                    actions.append('Practice stress management techniques such as mindful meditation or deep breathing 4-7-8')

            elif parameter == 'Calmness':
                if player_value < 5:
                    direct_assoc.append('Anxiety is associated to stress')
                    indirect_assoc.append('Anxiety can affect recovery and increase muscle soreness')
                    actions.append('Practice mindful meditation 10-20 min per day, adopt deep breathing 4-7-8 or consult a psychotherapist')

            elif parameter == 'Anger':
                if player_value > 5:
                    direct_assoc.append('Anger increases stress and affects sleep')
                    indirect_assoc.append('Anger reduces attention span and affects performance')
                    actions.append('Practice mindful meditation, adopt deep breathing 4-7-8 or consult a psychotherapist')

            elif parameter == 'Muscle Soreness':
                if player_value > 5:
                    direct_assoc.append('Muscle Soreness negatively impacts energy and sleep')
                    indirect_assoc.append('Increases stress and affects recovery')
                    actions.append('After game or training practice cold water immersions 5x1 min at 11*C, or 15 min at 15*C; practice muscle relaxation techniques via massage coupled with deep breathing 4-7-8; adopt a high protein diet (34%); hydrate with carbohydrate electrolytes and\or use intake 6g\day of L-alanyl-L-glutamine; use compression garments')

            elif parameter == 'Salivary pH':
                if player_value < 6.5 :
                    direct_assoc.append('Abnormal salivary pH')
                    indirect_assoc.append('Impacts hydration and energy levels')
                    actions.append('Ensure hydration with carbohydrate electrolytes and improve oral hygiene')
                elif player_value > 7.5:
                    direct_assoc.append('Abnormal salivary pH')
                    indirect_assoc.append('Impacts hydration and energy levels')
                    actions.append('Ensure hydration with carbohydrate electrolytes')

            elif parameter == 'Salivary Uric Acid mg/dl':
                if player_value > 7:
                    direct_assoc.append('High uric acid impacts sleep quality and cardiovascular health')
                    indirect_assoc.append('May contribute to inflammation and oxidative stress')
                    actions.append('Adopt a low-purine balanced diet and increase hydration with carbohydrate electrolytes')
                elif 0 < player_value < 2.5:
                    direct_assoc.append('Low uric acid indicates muscle soreness and reduced recovery')
                    indirect_assoc.append('Associated with higher blood lactate levels')
                    actions.append('Focus on muscular recovery strategies such as cold water immersions 5x1 min at 11*C, or 15 min at 15*C; practice muscle relaxation techniques via massage coupled with deep breathing 4-7-8; hydrate with carbohydrate electrolytes and\or use intake 6g\day of L-alanyl-L-glutamine; use compression garments; adopt a high protein diet (34%) and monitor lactate')

            elif parameter == 'Salivary Nitrates':
                if 0 < player_value < 2:
                    direct_assoc.append('Low nitrate levels impact cardiovascular tone')
                    indirect_assoc.append('Increase muscle soreness and slow recovery')
                    actions.append('Increase dietary nitrate intake (e.g., leafy greens, beets, pomegranade)')

            elif parameter == 'Ketones ng/ml':
                if player_value > 6:
                    direct_assoc.append('High ketone levels may indicate a hypoglycemic state')
                    indirect_assoc.append('Can reduce energy levels and cause tiredness')
                    actions.append('Seek nutritional food such as bananas or coconut bars')

            elif parameter == 'Traces Blood Saliva':
                if player_value > 0:
                    direct_assoc.append('Presence indicates inflammation or trauma')
                    actions.append('Consult a medical professional for evaluation')

            elif parameter == 'Cortisol nmol/L':
                if player_value > 26:
                    direct_assoc.append('High cortisol indicates stress response')
                    indirect_assoc.append('Impacts recovery and testosterone levels')
                    actions.append('Practice stress management techniques such as mindful meditation 10-20 min per day and\or deep breathing 4-7-8')
                elif 0 < player_value < 3:
                    direct_assoc.append('Low cortisol indicates adrenal insufficiency')
                    actions.append('Seek medical evaluation')

            elif parameter == 'Testosterone nmol/L':
                if 0 < player_value < 0.2:
                    direct_assoc.append('Low testosterone impacts muscle mass and recovery')
                    indirect_assoc.append('May result from high cortisol or poor sleep')
                    actions.append('Improve physical recovery; increase the protein intake to 34% daily; use supplementation with 0.17g\kg BCAA (2:1:1) and 0.04g per kg arginine and monitor hormone levels')

            elif parameter == 'TCR':
                if 0 < player_value < 0.04:
                    direct_assoc.append('Low TCR indicates poor anabolic balance')
                    indirect_assoc.append('Linked to high stress and inadequate recovery')
                    actions.append('Focus on physical recovery via high-protein nutrition (34%); use supplementation with 0.17g\kg BCAA (2:1:1) and 0.04g per kg arginine, and reduce stress via as mindful meditation 10-20 min per day and\or deep breathing 4-7-8')

            # Special handling for 0 values in key biomarkers
            if is_key_biomarker and player_value == 0:
                direct_assoc.append('Out of Range Value')
                indirect_assoc.append('Out of Range Value')
                actions.append('Redo Analysis')

            results.append({
                'Player': player_name,
                'Parameter': parameter,
                'Player Value': player_value,
                'Deviation from Average': deviation,
                'Date': date,
                'Direct Associations': ', '.join(direct_assoc) if direct_assoc else None,
                'Indirect Associations': ', '.join(indirect_assoc) if indirect_assoc else None,
                'Actions': ', '.join(actions) if actions else None
            })
    return pd.DataFrame(results)



def evaluate_single_player_data(player_values, player_name, analysis_date, diet_info):
    Diet = diet_info.get(player_name, '')
    results = []

    # Benchmark data
    mybenchmark_data = {
        'Data': [
            'Sleep quality', 'How many hours did you sleep?', 'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level',
            'Calmness', 'Anger', 'Muscle Soreness', 'SalivaCoding', 'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl',
            'Traces Blood Saliva', 'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L', 'Testosterone pg/ml', 'Testosterone nmol/L',
            'TCR', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score',
            'Biomarker_Factor', 'Total_Score', 'AST U/L', 'CK U/L', 'DHEA-S ng/L', 'DHEA-S nmol/L'
        ],
        'Benchmark Mean': [
            8, None, 1, 1, 8, 1, 10, 1, 1, 2025012913, 7, 2, 5, 0.5, 1.5, 8, 22.08, 300, 1.041, 0.047, 82.5, 100, 100, 100, 100, 100, 1, 99, 30, 50, 5, 13.55
        ],
        'Benchmark Min': [
            7, None, 0.8, 0.8, 6, 0.8, 8, 0.8, 0.8, 2025012913, 6, 1, 2, 0, 0, 4, 14, 150, 0, 0, 64, 50, 50, 50, 50, 50, 0.5, 49.5, 15, 25, 2.5, 6.775
        ],
        'Benchmark Max': [
            9, None, 1.2, 1.2, 10, 1.2, 12, 1.2, 1.2, 2025012913, 8, 3, 7, 1, 3, 12, 30, 450, 2, 0.11, 100, 150, 150, 150, 150, 150, 1.5, 150, 58.5, 97.5, 9.75, 26.4225
        ]
    }

    mybenchmark_df = pd.DataFrame(mybenchmark_data)

    key_biomarkers = [
        'Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Traces Blood Saliva',
        'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L',
        'Testosterone pg/ml', 'Testosterone nmol/L', 'TCR'
    ]

    for _, benchmark_row in mybenchmark_df.iterrows():
        parameter = benchmark_row['Data']

        if parameter not in player_values:
            continue

        player_value = player_values.get(parameter, 0 if parameter in key_biomarkers else np.nan)
        is_key_biomarker = parameter in key_biomarkers

        if pd.isna(player_value) and not is_key_biomarker:
            continue

        benchmark_mean = float(benchmark_row['Benchmark Mean']) if pd.notna(benchmark_row['Benchmark Mean']) else np.nan

        deviation_from_benchmark = (
            (player_value - benchmark_mean) / benchmark_mean
            if pd.notna(benchmark_mean) and benchmark_mean != 0 and pd.notna(player_value)
            else np.nan
        )

        direct_assoc = []
        indirect_assoc = []
        actions = []

        #
        if parameter == 'Sleep quality':
            if player_value < 6:
                direct_assoc.append('Poor sleep quality')
                indirect_assoc.append('Impacts tiredness, energy level, and stress')
                actions.append('Improve sleep hygiene by avoiding digital media browsing during night time and minimize light sources in the bedroom')

        elif parameter == 'Sleep Time Score':
            if player_value < 1:
                direct_assoc.append('Insufficient sleep duration')
                indirect_assoc.append('Affects tiredness and muscle soreness')
                actions.append('Aim for consistent 7-9 hours sleep')

        elif parameter == 'Tiredness':
            if player_value > 6:
                direct_assoc.append('High tiredness reflects poor recovery')
                indirect_assoc.append('Linked to poor sleep and impacts energy')
                actions.append('Prioritize recovery increasing sleep time up to 10 hours, improve sleep by decreasing muscular soreness through cold water immersions 5 x 1 min at 11*C, or 15 min at 15*C')

        elif parameter == 'Energy level':
            if player_value < 6:
                direct_assoc.append('Low energy')
                indirect_assoc.append('Impacted by stress, sleep, and recovery')
                actions.append('Improve diet by intaking 6g per day of L-alanyl-L-glutamine, 0.17g per kg BCAA (2:1:1) and 0.04g per kg arginine and manage stress through mindful meditation 10-20min per day')

        elif parameter == 'Stress level':
            if player_value > 5:
                direct_assoc.append('High stress negatively impacts energy')
                indirect_assoc.append('Increases muscle soreness and affects recovery')
                actions.append('Practice stress management techniques such as mindful meditation or deep breathing 4-7-8')

        elif parameter == 'Calmness':
            if player_value < 5:
                direct_assoc.append('Anxiety is associated to stress')
                indirect_assoc.append('Anxiety can affect recovery and increase muscle soreness')
                actions.append('Practice mindful meditation 10-20 min per day, adopt deep breathing 4-7-8 or consult a psychotherapist')

        elif parameter == 'Gloominess':
            if player_value > 5:
                direct_assoc.append('Low mood may affect energy and motivation')
                indirect_assoc.append('Can reduce performance quality and slow recovery')
                actions.append('Stay active, connect socially, and follow a steady routine')

        elif parameter == 'Anger':
            if player_value > 5:
                direct_assoc.append('Anger increases stress and affects sleep')
                indirect_assoc.append('Anger reduces attention span and affects performance')
                actions.append('Practice mindful meditation, adopt deep breathing 4-7-8 or consult a psychotherapist')

        elif parameter == 'Muscle Soreness':
            if player_value > 5:
                direct_assoc.append('Muscle Soreness negatively impacts energy and sleep')
                indirect_assoc.append('Increases stress and affects recovery')
                actions.append('After game or training practice cold water immersions 5x1 min at 11*C, or 15 min at 15*C; practice muscle relaxation techniques via massage coupled with deep breathing 4-7-8; adopt a high protein diet (34%); hydrate with carbohydrate electrolytes and or use intake 6g per day of L-alanyl-L-glutamine; use compression garments')

        elif parameter == 'Salivary pH':
            if player_value < 6.5:
                direct_assoc.append('Abnormal salivary pH')
                indirect_assoc.append('Impacts hydration and energy levels')
                actions.append('Ensure hydration with carbohydrate electrolytes and improve oral hygiene')
            elif player_value > 7.5:
                direct_assoc.append('Abnormal salivary pH')
                indirect_assoc.append('Impacts hydration and energy levels')
                actions.append('Ensure hydration with carbohydrate electrolytes')

        elif parameter == 'Salivary Uric Acid mg/dl':
            if player_value > 7:
                direct_assoc.append('High uric acid impacts sleep quality and cardiovascular health')
                indirect_assoc.append('May contribute to inflammation and oxidative stress')
                actions.append('Adopt a low-purine balanced diet and increase hydration with carbohydrate electrolytes')
            elif 0 < player_value < 2.5:
                direct_assoc.append('Low uric acid indicates muscle soreness and reduced recovery')
                indirect_assoc.append('Associated with higher blood lactate levels')
                actions.append('Focus on muscular recovery strategies such as cold water immersions 5x1 min at 11*C, or 15 min at 15*C; practice muscle relaxation techniques via massage coupled with deep breathing 4-7-8; hydrate with carbohydrate electrolytes or use intake 6g per day of L-alanyl-L-glutamine; use compression garments; adopt a high protein diet (34%) and monitor lactate')

        elif parameter == 'Salivary Nitrates':
            if 0 < player_value < 2:
                direct_assoc.append('Low nitrate levels impact cardiovascular tone')
                indirect_assoc.append('Increase muscle soreness and slow recovery')
                actions.append('Increase dietary nitrate intake (e.g., leafy greens, beets, pomegranate)')

        elif parameter == 'Ketones ng/ml':
            if player_value > 6:
                direct_assoc.append('High ketone levels may indicate a hypoglycemic state')
                indirect_assoc.append('Can reduce energy levels and cause tiredness')
                actions.append('Seek nutritional food such as bananas or coconut bars')
            elif  player_value < 2 and Diet =='High Carbs':
                actions.append('reduce intake of sugar-based food as you are overfueling')

            elif player_value > 8 and Diet== 'High Protein':
                actions.append('reduce intake of Proteins as you are overloading your kidney function')

            elif player_value > 8  and   Diet in ['Balanced', 'Vegan', 'Vegetarian']:
                actions.append('It is time to fuel up, you are using internal energy stores')

        elif parameter == 'Traces Blood Saliva':
            if player_value > 0:
                direct_assoc.append('Presence indicates inflammation or trauma')
                actions.append('Consult a medical professional for evaluation')

        elif parameter == 'Cortisol nmol/L':
            if player_value > 26:
                direct_assoc.append('High cortisol indicates stress response')
                indirect_assoc.append('Impacts recovery and testosterone levels')
                actions.append('Practice stress management techniques such as mindful meditation 10-20 min per day and or deep breathing 4-7-8')
            elif 0 < player_value < 3:
                direct_assoc.append('Low cortisol indicates adrenal insufficiency')
                actions.append('Seek medical evaluation')

        elif parameter == 'Testosterone nmol/L':
            if 0 < player_value < 0.2:
                direct_assoc.append('Low testosterone impacts muscle mass and recovery')
                indirect_assoc.append('May result from high cortisol or poor sleep')
                actions.append('Improve physical recovery; increase the protein intake to 34% daily; use supplementation with 0.17g per kg BCAA (2:1:1) and 0.04g per kg arginine and monitor hormone levels')

        elif parameter == 'TCR':
            if 0 < player_value < 0.04:
                direct_assoc.append('Low TCR indicates poor anabolic balance')
                indirect_assoc.append('Linked to high stress and inadequate recovery')
                actions.append('Focus on physical recovery via high-protein nutrition (34%); use supplementation with 0.17g per kg BCAA (2:1:1) and 0.04g per kg arginine, and reduce stress via mindful meditation 10-20 min per day and or  deep breathing 4-7-8')

        # Special handling for 0 values in key biomarkers
        if is_key_biomarker and player_value == 0:
            direct_assoc.append('Out of Range Value or Analyte not tested')
            indirect_assoc.append('Out of Range Value or Analyte not tested')
            #actions.append('Redo Analysis')


        results.append({
            'Player': player_name,
            'Parameter': parameter,
            'Player Value': player_value,
            'Benchmark Mean': benchmark_mean,
            'Deviation from Benchmark': deviation_from_benchmark,
            'Deviation from Average': deviation_from_benchmark,
            'Date': analysis_date,
            'Direct Associations': ', '.join(direct_assoc) if direct_assoc else None,
            'Indirect Associations': ', '.join(indirect_assoc) if indirect_assoc else None,
            'Actions': ', '.join(actions) if actions else None
        })

    return  pd.DataFrame(results)


################################################
@app.route('/select_player', methods=['GET', 'POST'])
@login_required
def select_player():
    if request.method == 'POST':
        player_id = request.form.get('player_id')
        return redirect(url_for('player_report', player_id=player_id))

    # Read the CSV file
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Pallacanestro_Trieste_Season_2_2025.csv')
    data = pd.read_csv(csv_path)
    unique_players = data[data['Player'] != 1000]['Player'].unique()

    return render_template('select_player.html', players=unique_players)

@app.route('/player_report/<int:player_id>')
@login_required
def player_report(player_id):
    # Read the CSV file
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'Pallacanestro_Trieste_Season_2_2025.csv')
    evaluation_path = os.path.join(app.config['PLOT_FOLDER'], 'Player_Evaluation_Results_With_Deviation_And_Date.csv')

    data, reference_row, cols_to_normalize = prepare_radar_data(csv_path)

    # Generate Radar Plot
    radar_plot = generate_radar_plot(data, player_id, reference_row, cols_to_normalize)

    # Generate Timeline Plot
    data = load_and_prepare_timeline_data(csv_path)
    timeline_plot = generate_timeline_plot(data, player_id)

    ########Score card
    df = pd.read_csv(evaluation_path)
    df['Player Value'] = pd.to_numeric(df['Player Value'], errors='coerce')
    df['Deviation from Average'] = pd.to_numeric(df['Deviation from Average'], errors='coerce')

    # Process the evaluation data

    benchmark_data = {
        'Data': [
            'Sleep quality', 'How many hours did you sleep?', 'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level',
            'Calmness', 'Anger', 'Muscle Soreness', 'SalivaCoding', 'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl',
            'Traces Blood Saliva', 'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L', 'Testosterone pg/ml', 'Testosterone nmol/L',
            'TCR', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score',
            'Biomarker_Factor', 'Total_Score', 'AST U/L', 'CK U/L', 'DHEA-S ng/L', 'DHEA-S nmol/L'
        ],
        'Benchmark Mean': [
            8, None, 1, 1, 8, 1, 10, 1, 1, 2025012913, 7, 2, 5, 0.5, 1.5, 8, 22.08, 300, 1.041, 0.047, 82.5, 100, 100, 100, 100, 100, 1, 99, 30, 50, 5, 13.55
        ],
        'Benchmark Min': [
            7, None, 0.8, 0.8, 6, 0.8, 8, 0.8, 0.8, 2025012913, 6, 1, 2, 0, 0, 4, 14, 150, 0, 0, 64, 50, 50, 50, 50, 50, 0.5, 49.5, 15, 25, 2.5, 6.775
        ],
        'Benchmark Max': [
            9, None, 1.2, 1.2, 10, 1.2, 12, 1.2, 1.2, 2025012913, 8, 3, 7, 1, 3, 12, 30, 450, 2, 0.11, 100, 150, 150, 150, 150, 150, 1.5, 150, 58.5, 97.5, 9.75, 26.4225
        ]
    }
    benchmark_df = pd.DataFrame(benchmark_data)
    df = df[['Player', 'Parameter', 'Player Value', 'Deviation from Average', 'Date', 'Direct Associations', 'Actions']]
    df = df.dropna(subset=['Player', 'Parameter', 'Player Value'])

    specified_metrics = [
            'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Ketones ng/ml',
            'Cortisol nmol/L', 'Testosterone nmol/L', 'TCR', 'Biomarker_Score', 'Total_Score', 'Self_Reported_Score'
        ]
    df = df[df['Parameter'].isin(specified_metrics)]

    # Group by player
    players_data = df.groupby('Player') #may need for fix
    unique_players = df['Player'].unique()

    player = int(player_id)
    data = df[df['Player'] == player_id]
    metrics_html = ""
    total_score = 0
    biomarker_score = 0
    self_reported_score = 0
    total_score_change = 0
    biomarker_score_change = 0
    self_reported_score_change = 0
    # Retrieve all actions for the player
    actions_row = data[data['Actions'].notna() & (data['Actions'].str.strip() != "")]
    if not actions_row.empty:
        # Get all actions as a single string joined by slashes
        all_actions = " / ".join(actions_row['Actions'].dropna().astype(str).unique())

        # Split the string by slashes to create individual action items
        action_items = [item.strip() for item in all_actions.split('/') if item.strip()]

        # Create HTML bullet list with center alignment
        if action_items:
            actions_html = "<ul style='list-style-type: disc; margin: 0 auto; padding: 0; text-align: justify; list-style-position: inside;'>"
            for item in action_items:
                actions_html += f"<li style='display: block; margin-bottom: 5px; text-align: justify;'>{item}</li>"
            actions_html += "</ul>"
            actions = actions_html
        else:
            actions = "<div style='text-align: justify;'>No actions specified</div>"
    else:
        actions = "<div style='text-align: justify;'>No actions specified</div>"

    #Handle self-reported scores for this player
    self_reported_rows = data[data['Parameter'] == 'Self_Reported_Score']
    if not self_reported_rows.empty:
        latest_date = self_reported_rows['Date'].max()
        latest_score_row = self_reported_rows[self_reported_rows['Date'] == latest_date]

        if not latest_score_row.empty:
            self_reported_score = round(latest_score_row['Player Value'].values[0], 2)
            self_reported_score_change = round(latest_score_row['Deviation from Average'].values[0] * 100, 2)

    # Process metrics for this player
    for index, row in data.iterrows():
        if row['Parameter'] == 'Total_Score' and pd.notna(row['Player Value']):
            total_score = round(row['Player Value'], 2)
            total_score_change = round(row['Deviation from Average'] * 100, 2) if pd.notna(row['Deviation from Average']) else 0
        elif row['Parameter'] == 'Biomarker_Score' and pd.notna(row['Player Value']):
            biomarker_score = round(row['Player Value'], 2)
            biomarker_score_change = round(row['Deviation from Average'] * 100, 2) if pd.notna(row['Deviation from Average']) else 0

    # Determine the color for the scores
    total_score_color = "#4caf50" if total_score > 50 else "#f44336"
    biomarker_score_color = "#4caf50" if biomarker_score > 50 else "#f44336"
    self_reported_score_color = "#4caf50" if self_reported_score > 50 else "#f44336"

    # Determine the change class and icon for scores
    total_score_change_class = "up" if total_score_change > 0 else "down"
    total_score_change_icon = "🔼" if total_score_change > 0 else "🔽"
    biomarker_score_change_class = "up" if biomarker_score_change > 0 else "down"
    biomarker_score_change_icon = "🔼" if biomarker_score_change > 0 else "🔽"
    self_reported_score_change_class = "up" if self_reported_score_change > 0 else "down"
    self_reported_score_change_icon = "🔼" if self_reported_score_change > 0 else "🔽"

    # Create organized metrics for left and right columns
    left_metrics_html = ""
    right_metrics_html = ""

    # Define the order for left and right columns
    left_metrics_order = ['Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl', 'Ketones ng/ml']
    right_metrics_order = ['Cortisol nmol/L', 'Testosterone nmol/L', 'TCR']

    # Process metrics for left column (pH, Nitrates, Uric Acid, Ketones)
    for metric_name in left_metrics_order:
        metric_rows = data[data['Parameter'] == metric_name]
        if not metric_rows.empty:
            row = metric_rows.iloc[0]  # Take the first row for this metric
            if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
                benchmark_row = benchmark_df[benchmark_df['Data'] == metric_name]
                if not benchmark_row.empty:
                    benchmark_min = benchmark_row['Benchmark Min'].values[0]
                    benchmark_max = benchmark_row['Benchmark Max'].values[0]
                    benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
                else:
                    print(f"WARNING: No benchmark found for {metric_name}")
                    benchmark_min = None
                    benchmark_max = None
                    benchmark_mean = None

                direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

                left_metrics_html += generate_metric_html(
                    metric_name, row['Player Value'], row['Deviation from Average'],
                    benchmark_min, benchmark_max, benchmark_mean, direct_association
                )

    # Process metrics for right column (Cortisol, Testosterone)
    for metric_name in right_metrics_order[:2]:  # Only process Cortisol and Testosterone here
        metric_rows = data[data['Parameter'] == metric_name]
        if not metric_rows.empty:
            row = metric_rows.iloc[0]  # Take the first row for this metric
            if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
                benchmark_row = benchmark_df[benchmark_df['Data'] == metric_name]
                if not benchmark_row.empty:
                    benchmark_min = benchmark_row['Benchmark Min'].values[0]
                    benchmark_max = benchmark_row['Benchmark Max'].values[0]
                    benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
                else:
                    print(f"WARNING: No benchmark found for {metric_name}")
                    benchmark_min = None
                    benchmark_max = None
                    benchmark_mean = None

                direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

                right_metrics_html += generate_metric_html(
                    metric_name, row['Player Value'], row['Deviation from Average'],
                    benchmark_min, benchmark_max, benchmark_mean, direct_association
                )

    # Process TCR separately
    tcr_rows = data[data['Parameter'] == 'TCR']
    if not tcr_rows.empty:
        row = tcr_rows.iloc[0]
        if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
            benchmark_row = benchmark_df[benchmark_df['Data'] == 'TCR']
            if not benchmark_row.empty:
                benchmark_min = benchmark_row['Benchmark Min'].values[0]
                benchmark_max = benchmark_row['Benchmark Max'].values[0]
                benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
            else:
                print(f"WARNING: No benchmark found for TCR")
                benchmark_min = None
                benchmark_max = None
                benchmark_mean = None

            direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

            # Add TCR with actions at the bottom of the right column
            right_metrics_html += generate_tcr_metric_html(
                row['Player Value'], row['Deviation from Average'],
                benchmark_min, benchmark_max, benchmark_mean, direct_association, actions
            )
    else:
        print(f"WARNING: No TCR data found for Player {int(player)}")


    return render_template('player_report.html',
                         player_id=player_id,
                         radar_plot=radar_plot,
                         timeline_plot=timeline_plot,
                         total_score=total_score,
                         total_score_change=abs(total_score_change),  # Use absolute value for display
                        total_score_change_class=total_score_change_class,
#                        total_score_change_icon=total_score_change_icon,
                        biomarker_score=biomarker_score,
                        biomarker_score_change=abs(biomarker_score_change),  # Use absolute value for display
                        biomarker_score_change_class=biomarker_score_change_class,
#                        biomarker_score_change_icon=biomarker_score_change_icon,
                        self_reported_score=self_reported_score,
                        self_reported_score_change=abs(self_reported_score_change),  # Use absolute value for display
                        self_reported_score_change_class=self_reported_score_change_class,
#                        self_reported_score_change_icon=self_reported_score_change_icon,
                        total_score_color=total_score_color,
                        biomarker_score_color=biomarker_score_color,
                        self_reported_score_color=self_reported_score_color,
                        left_metrics_html=left_metrics_html,
                        right_metrics_html=right_metrics_html,
                        current_year=datetime.now().year)#
                         #scores_html=scores_html,
                         #metrics_html=metrics_html)


def generate_radar_plot(data, player_id, reference_row, cols_to_normalize):
    # Get last two dates for the player
    player_dates = data[data['Player'] == player_id]['Date'].sort_values(ascending=False).unique()

    if len(player_dates) >= 2:
        date_1 = player_dates[0]
        date_2 = player_dates[1]
    elif len(player_dates) == 1:
        date_1 = player_dates[0]
        date_2 = None
    else:
        date_1 = None
        date_2 = None

    # Create radar plot
    fig = go.Figure()

    # Add traces for each date
    if date_1 is not None:
        player_data_1 = data[(data['Player'] == player_id) & (data['Date'] == date_1)]
        if not player_data_1.empty:
            values_1 = player_data_1[cols_to_normalize].iloc[0].tolist()
            fig.add_trace(go.Scatterpolar(
                r=values_1 + [values_1[0]],
                theta=cols_to_normalize + [cols_to_normalize[0]],
                fill='toself',
                name=date_1.strftime('%m/%d/%Y'),
                #name = datetime.strptime(date_1, '%Y-%m-%d').strftime('%m/%d/%Y'),
                line=dict(color=RED),
                opacity=0.8
            ))

    if date_2 is not None:
        player_data_2 = data[(data['Player'] == player_id) & (data['Date'] == date_2)]
        if not player_data_2.empty:
            values_2 = player_data_2[cols_to_normalize].iloc[0].tolist()
            fig.add_trace(go.Scatterpolar(
                r=values_2 + [values_2[0]],
                theta=cols_to_normalize + [cols_to_normalize[0]],
                fill='toself',
                name=date_2.strftime('%m/%d/%Y'),
                line=dict(color=DARK_BLUE),
                opacity=0.8
            ))


    #reference_data = pd.Series(reference_row[cols_to_normalize])
    #reference_data =  reference_row[cols_to_normalize].iloc[0]
    #ref_values = reference_data.tolist()
    #ref_values += ref_values[:1]
    # Add reference trace
    ref_values = reference_row[cols_to_normalize].tolist()

    fig.add_trace(go.Scatterpolar(
        r=ref_values + [ref_values[0]],
        theta=cols_to_normalize + [cols_to_normalize[0]],
        fill='toself',
        name='Reference',
        line=dict(color=GOLD),
        opacity=0.5
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
               #range=[0, 100]
            )
        ),
        #title=f'Player {player_id} Self-reported metrics',
        font=dict(size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return pio.to_html(fig, full_html=False, config={'staticPlot': True})



def generate_timeline_plot(data, player_id):
    player_data = data[data['Player'] == player_id].sort_values('Date')
    if player_data.empty:
        return None
    # Handle duplicate dates by averaging

    fig = go.Figure()

    # Add traces for each score type
    score_types = [
        ('Self_Reported_Score', 'Self Report Score', DARK_BLUE),
        ('Biomarker_Score', 'Biomarker Score', RED),
        ('Total_Score', 'Total Score', GOLD)
    ]

    for score_type, label, color in score_types:
        if score_type in player_data.columns:
            valid_data = player_data[['Date', score_type]].dropna()
            if not valid_data.empty:
                fig.add_trace(go.Scatter(
                    x=valid_data['Date'],
                    y=valid_data[score_type],
                    name=label,
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=8)
                ))


    # X-axis ticks every 15 days
    min_date = player_data['Date'].min()
    max_date = player_data['Date'].max()
    date_range = (max_date - min_date).days
    padding_days = max(10, int(date_range * 0.1))
    padded_start = min_date - timedelta(days=padding_days)
    padded_end = max_date + timedelta(days=padding_days)

    tick_dates = [min_date]
    current_date = min_date
    while current_date <= max_date:
        current_date += timedelta(days=15)
        tick_dates.append(current_date)
    # Update layout
    fig.update_layout(
        #title=dict(text=f'Score Trends for Player {player_id}', font=dict(size=20, color=DARK_BLUE), x=0.5),
        xaxis=dict(
            title='Date',
            titlefont=dict(size=14),
            tickformat='%d/%m',
            tickvals=tick_dates,
            range=[padded_start, padded_end],
            tickfont=dict(size=14, color='black'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linecolor='black',
            mirror=True,
            showspikes=True,
            spikemode='across'
        ),
        yaxis=dict(
            title='Score',
            titlefont=dict(size=14),
            tickfont=dict(size=14, color='black'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linecolor='black',
            mirror=True
        ),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            font=dict(size=12, color='black'),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=0.5,
            x=0.01,
            y=0.01,
            xanchor='left',
            yanchor='bottom'
        ),
        hovermode='x unified'
    )
    return pio.to_html(fig, full_html=False, config={'staticPlot': True})

def generate_metric_html(metric_name, value, deviation, benchmark_min, benchmark_max, benchmark_mean, direct_association):
    value_rounded = round(value, 2)
    deviation_rounded = round(deviation * 100, 2)
    deviation_class = "up" if deviation_rounded > 0 else "down"
    deviation_icon = "🔼" if deviation_rounded > 0 else "🔽"
    deviation_formatted = f"+{deviation_rounded}%" if deviation_rounded > 0 else f"{deviation_rounded}%"
    metric_icons = {
    "Salivary pH": "⚖️",
    "Salivary Nitrates": "❤️",
    "Salivary Uric Acid mg/dl": "🔄",
    "Ketones ng/ml": "🧊",
    "Cortisol nmol/L": "⚡",
    "Testosterone nmol/L": "💪",
    "TCR": "🧠",
    }
    metric_icon = metric_icons.get(metric_name, "")
    if benchmark_max == benchmark_min:
        player_position = 50
    else:
        clamped_value = max(benchmark_min, min(benchmark_max, value))
        player_position = ((clamped_value - benchmark_min) / (benchmark_max - benchmark_min)) * 100
    note_html = ""
    if pd.notna(direct_association) and direct_association.strip():
        note_html = f'<div class="metric-note">{direct_association}</div>'
    metric_bar_class = "metric-bar nitrate" if metric_name == "Salivary Nitrates" else "metric-bar"
    return f'''
    <div class="metric">
        <div>{metric_icon} {metric_name}: {value_rounded}</div>
        <div class="{metric_bar_class}">
            <div class="metric-line" style="left: {player_position}%;"></div>
        </div>
   <!--   <div class="metric-change {deviation_class}">{deviation_icon} {deviation_formatted}</div>  -->
        {note_html}
    </div>
    '''

# Function to generate TCR metric with the Actions field below it
def generate_tcr_metric_html(value, deviation, benchmark_min, benchmark_max, benchmark_mean, direct_association, actions):
    value_rounded = round(value, 2)
    deviation_rounded = round(deviation * 100, 2)
    deviation_class = "up" if deviation_rounded > 0 else "down"
    deviation_icon = "🔼" if deviation_rounded > 0 else "🔽"
    deviation_formatted = f"+{deviation_rounded}%" if deviation_rounded > 0 else f"{deviation_rounded}%"
    if benchmark_max == benchmark_min:
        player_position = 50
    else:
        clamped_value = max(benchmark_min, min(benchmark_max, value))
        player_position = ((clamped_value - benchmark_min) / (benchmark_max - benchmark_min)) * 100
    note_html = ""
    if pd.notna(direct_association) and direct_association.strip():
        note_html = f'<div class="metric-note">{direct_association}</div>'

    return f'''
    <div class="metric">
        <div>🧠 TCR: {value_rounded}</div>
        <div class="metric-bar">
            <div class="metric-line" style="left: {player_position}%;"></div>
        </div>
        <div class="metric-change {deviation_class}">{deviation_icon} {deviation_formatted}</div>
        {note_html}
    </div>
    <div style="margin-top: 15px;">
        <div class="footer-actions">
            <div class="footer-actions-content">
                <div class="actions-title" style="text-align: center;">ACTIONS:</div>
                {actions}
            </div>
        </div>
    </div>
    '''


##############################################################
@app.route('/show_sequence/<int:page_index>')
@login_required
def show_sequence(page_index):
    html_pages = [
        'all_players_dashboard.html',
        'basketball_performance_landscape.html',
        'performance_classification_final.html',
        'Players_A4_Beveled_Scorecards_10pt.html',
        'all_players_multivariates.html',  # add more if needed
        'performance_classification_enhanced.html',
        'causal_interaction_analysis_report.html']

    if 0 <= page_index < len(html_pages):
        current_page = html_pages[page_index]
        next_index = page_index + 1 if page_index + 1 < len(html_pages) else None
        return render_template(current_page, next_index=next_index)
    else:
        flash('No more pages.', 'info')
        return redirect(url_for('dashboard'))  # or wherever you want to go at the end




@app.route('/plot', methods=['GET', 'POST'])
@login_required
def plot():
    # Initialize variables
    selected_date = None
    plot_html = plot_html2 = plot_html3 = None
    existing_files = get_existing_files()
    available_dates = []  # Initialize empty list
    selected_file = None
    current_filename = None

    # Handle GET request - show initial file selection
    if request.method == 'GET':
        return render_template('dashboard.html',
                            existing_files=existing_files,
                            show_athletes_section=True)

    # Handle POST requests (both steps)
    if request.method == 'POST':
        # STEP 1: File selection
        if 'existing_file' in request.form and 'selected_date' not in request.form:
            filename = request.form['existing_file']

            if not filename:
                flash('Please select a file', 'error')
                return render_template('dashboard.html',
                                    existing_files=existing_files,
                                    show_athletes_section=True)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


            try:
                # Load and process the file to get available dates
                data = pd.read_csv(filepath)
                data.rename(columns={
                    'Sleep Time Score_x': 'Sleep Time Score',
                    'Performance_x': 'Performance',
                    'Date_x': 'Date'
                }, inplace=True)

                data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
                available_dates = [d.strftime('%d/%m/%Y') for d in sorted(data['Date'].unique())]
                #available_dates = sorted(data['Date'].dt.strftime('%d/%m/%Y').unique())

                return render_template('dashboard.html',
                                    existing_files=existing_files,
                                    selected_file=filename,
                                    available_dates=available_dates,
                                    show_athletes_section=True)
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return render_template('dashboard.html',
                                    existing_files=existing_files,
                                    show_athletes_section=True)

        # STEP 2: Date selection and plot generation
        elif 'selected_date' in request.form:
            filename = request.form['existing_file']
            selected_date_str = request.form['selected_date']

            if not selected_date_str:
                flash('Please select a date', 'error')
                return redirect(url_for('plot'))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                # Process the data with selected date
                data = pd.read_csv(filepath)
                ### fixixng for recovery date data only
                #data = data[data['Day'] == 'Recovery']  # Keep only 'Recovery' rows
                data.rename(columns={
                    'Sleep Time Score_x': 'Sleep Time Score',
                    'Performance_x': 'Performance',
                    'Date_x': 'Date'
                }, inplace=True)

                data['Date'] = pd.to_datetime(data['Date'])

                # Check for any dates that failed to parse
                if data['Date'].isna().any():
                    print("Warning: Some dates couldn't be parsed. Sample problematic dates:")
                    print(data[data['Date'].isna()]['Date'].head())
                    data = data.dropna(subset=['Date'])

                data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
                selected_date = pd.to_datetime(selected_date_str, format='%d/%m/%Y')

                # Get dates of interest and loss dates
                dates_of_interest = get_dates_up_to_cutoff(data, cutoff_date=selected_date)
                #dates_of_interest = [
                #"01/14/2025", "01/22/2025", "01/29/2025",
                #"02/05/2025", "02/26/2025", "03/05/2025",
                #"03/12/2025", "03/19/2025", "03/26/2025",
                #"04/02/2025", "04/09/2025", "04/16/2025",
                #"04/23/2025", "04/30/2025", "05/07/2025",
                #"05/14/2025"
                #    ]
                loss_dates = get_match_loss_dates(data, selected_date)

                # Initialize dictionary to store the aggregate scores


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
        #exclusions = {
        #4: ["2025-02-26", "2025-03-05", "2025-03-12","2025-03-19","2025-03-26", "2025-04-02", "2025-04-09" ],
        #2: ["2025-02-26", "2025-03-05", "2025-03-12"],
        #11: ["2025-03-19","2025-03-26", "2025-04-02", "2025-04-09"],
        #1: ["2025-03-26", "2025-04-09","2025-04-16"]  # Added exclusions for Player 1
        #    }
                exclusions = {
                4: ["2/26/2025", "3/5/2025", "3/12/2025", "3/19/2025", "3/26/2025", "4/2/2025", "4/9/2025"],
                2: ["2/26/2025", "3/5/2025", "3/12/2025"],
                11: ["3/19/2025", "3/26/2025", "4/2/2025", "4/9/2025", "4/16/2025",  "4/23/2025", "4/30/2025"],
                1: ["3/26/2025", "4/9/2025"]#, "4/16/2025"]  # Added exclusions for Player 1
                }

                # Now parsing as m/d/Y works correctly
                for player in exclusions:
                    exclusions[player] = [pd.to_datetime(date, format='%m/%d/%Y') for date in exclusions[player]]


                # Convert exclusion dates to datetime
                for player in exclusions:
                    exclusions[player] = [pd.to_datetime(date, format='%m/%d/%Y') for date in exclusions[player]]

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

                fig = px.line(aggregate_df, x='Date', y=['Median_Self_Reported_Score', 'Median_Biomarker_Score', 'Median_Total_Score'],
                          color_discrete_map={
                            'Median_Self_Reported_Score': '#b21f1f',
                            'Median_Biomarker_Score': '#1a2a6c',
                            'Median_Total_Score': '#1e988a'
                         },
                        markers=True,
                        title='Aggregate Scores Timeline')

                # Update traces to match your matplotlib style
                fig.update_traces(connectgaps=True, line_shape='linear', marker=dict(size=8), line=dict(width=2))
                # Add vertical lines for match losses
                #loss_dates = [pd.to_datetime(date, format='%d/%m/%Y') for date in ["22/01/2025", "5/02/2025", "12/03/2025", "09/04/2025"]]
                for date in loss_dates:
                    fig.add_vline(x=date, line_dash="dot", line_color="red", line_width=2)#, opacity=0.7)

                # Update layout to match your matplotlib style

                fig.update_layout(xaxis_title='Date',
                yaxis_title='Median Score',
                legend_title='',
                font=dict(
                    size=12 ),
                title_font_size=20,
                title_font=dict(
                    weight='bold'
                 ),
                xaxis=dict(
                 tickfont=dict(size=10),
                   tickangle=-45
                ),
             yaxis=dict(
                 tickfont=dict(size=10)
             ),
                plot_bgcolor='white',
             hovermode='x unified',
                legend=dict(
                        x=0.5,                      # Center horizontally
                        y=0.95,                     # Near the top (95% from bottom)
                    xanchor='center',           # Center anchor point
                    yanchor='top',              # Top anchor point
                    orientation='h',            # Horizontal orientation (3 items in a row)
                font=dict(size=8),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=0
                    ),
                 #margin=dict(l=40, r=40, t=80, b=40),
                   autosize=True,
                width=None,
                height=None
                    )
                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')


                output_path = os.path.join(app.config['PLOT_FOLDER'], 'aggregateplot.png')
                fig.write_image(output_path)
                plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': False} )



                #### PCA plot
                #filtered_data = data[data['Day'].isin(['Post_game', 'Recovery'])]
                #cutoff_date=selected_date
                filtered_data = data[data['Day'].isin(['Recovery'])]

                ### fixing here
                #if cutoff_date is None:
                #    cutoff = data['Date'].max()
                #else:
                    # Force parsing as m/d/Y (month first)
                #    cutoff = pd.to_datetime(cutoff_date, format='%m/%d/%Y')
                #data['Date'] = pd.to_datetime(data['Date'])


                #filtered_data = data[(data['Day'] == 'Recovery') ]# & (data['Date'].isin(dates_of_interest))]

                #filtered_data = data[(data['Day'] == 'Recovery') & (data['Date'] <= cutoff)]


                ############
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
                principal_components = pca.fit_transform(scaled_data)#

                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(principal_components)
                pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = clusters
                pca_df['Player'] = player_ids.values
                cluster_colors = ['#1a2a6c', '#b21f1f', '#1e988a']  # Example: blue, red, yellow
                #cluster_colors = ['#0072B2', '#E69F00', '#009E73']

                #### The classification table from function
                classification_html = create_classification_table(pca_df, data, cluster_colors)

                fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                color_discrete_sequence=cluster_colors,
                hover_data=['Player'],
                title='Player Classification')#,
        #width=1200,
        #height=600
        #)
                fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                # Add cluster labels at centroids
                for cluster_id in range(3):
                    cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
                    centroid = cluster_data[['PC1', 'PC2']].mean()#

                    fig.add_annotation(
                        x=centroid['PC1'],
                        y=centroid['PC2'],
                        text=f"Cluster {cluster_id}",
                        showarrow=False,
                        font=dict(size=18, color='white'),
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                        bgcolor=cluster_colors[cluster_id],
                        opacity=0.85
                        )
                # Add player numbers as text
                #for i, row in pca_df.iterrows():
                #    fig.add_annotation(
                 #       x=row['PC1']+ 0.5,
                 #       y=row['PC2']+ 0.5,
                 #       text=str(int(row['Player'])),
                 #       showarrow=False,
                 #       font=dict(size=12, color='black'),
                 #       bgcolor='rgba(255,255,255,0.6)',
                  #      bordercolor='black',
                  #      borderwidth=0.5,
                  #      opacity=0.9
                  #      )

                # Update layout
                fig.update_layout(
                 xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                title_font_size=24,
                title_font=dict(
                weight='bold'
                        ),
                xaxis=dict(title_font=dict(size=20), tickfont=dict(size=16)),
                yaxis=dict(title_font=dict(size=20), tickfont=dict(size=16)),
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False
                    )

                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                # Save as HTML (for web app) or static image
                output_path = os.path.join(app.config['PLOT_FOLDER'], 'PCplot.png')
                fig.write_image(output_path)
                plot_html2 = fig.to_html(full_html=False)


                ##SHAP plot
                #filtered_data = data[data['Day'].isin(['Post_game', 'Recovery'])]
                filtered_data = data[data['Day'].isin(['Recovery'])]

                ###fixing here
                #if cutoff_date is None:
                #    cutoff = data['Date'].max()
                #else:
                    # Force parsing as m/d/Y (month first)
                #    cutoff = pd.to_datetime(cutoff_date, format='%m/%d/%Y')
                #data['Date'] = pd.to_datetime(data['Date'])
                ###

                #dates_of_interest = pd.to_datetime(dates_of_interest)
                #filtered_data = data[(data['Day'] == 'Recovery') & (data['Date'].isin(dates_of_interest))]


                #filtered_data = data[(data['Day'] == 'Recovery') & (data['Date'] <= cutoff)]
                #################


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


                model = RandomForestRegressor(random_state=42, max_depth=3)
                model.fit(X, y)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                plt.figure(figsize=(6, 2.5))
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                plt.title('Feature Importance (SHAP Values)', fontsize=10)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()

                plt.savefig(os.path.join(app.config['PLOT_FOLDER'], 'checkSHAPplot.png'))
                plt.close()
                feature_names = X.columns
                feature_importance = np.abs(shap_values).mean(0)
                sorted_idx = np.argsort(feature_importance)[::-1]

                # Prepare data
                sorted_features = [feature_names[i] for i in sorted_idx]
                sorted_values = feature_importance[sorted_idx]

                fig = go.Figure()

                # Add bars
                fig.add_trace(go.Bar(
                    y=sorted_features,
                    x=sorted_values,
                    orientation='h',
                    marker_color='#1a2a6c',
                    opacity=0.8,
                    hoverinfo='x',
                    hovertemplate='%{y}: %{x:.3f}<extra></extra>'
                    ))

                # Add value labels
                for i, v in enumerate(sorted_values):
                    fig.add_annotation(
                        x=v + 0.0001,
                        y=i,
                        text=f"{v:.3f}",
                        showarrow=False,
                        font=dict(size=15, color='black'), #, family="Arial Black"),
                        xanchor='left'
                        )

                # Update layout
                fig.update_layout(
                    title='Feature Importance (SHAP Values)',
                    xaxis_title='mean(|SHAP value|)',
                    margin=dict(l=150, r=50, b=50, t=100),
                    title_font_size=24,
                    title_font=dict(
                    weight='bold'
                    ),
                    xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18), showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(tickfont=dict(size=22), autorange="reversed"),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                    )
            #width=1200,
            #height=600,
        # Save as HTML (for web app) or static image

                output_path = os.path.join(app.config['PLOT_FOLDER'], 'SHAPplot.png')
                fig.write_image(output_path)
                plot_html3 = fig.to_html(full_html=False)
                #return render_template('dashboard.html',
                # get all players Radar and timeline plot and html for scre card


                return render_template('b_on_fly_coach.html',
                            plot_html=plot_html,
                            #plot_html= base64_str,
                            plot_html2=plot_html2,
                            plot_html3=plot_html3,
                            existing_files=get_existing_files(),
                            selected_file=filename,  # Required for the second dropdown
                            available_dates=available_dates,
                            selected_date=selected_date,  # Pass to tem
                            current_filename=filename,
                            classification_table=classification_html
                            )


# uncomment below if update of user is working


#        return render_template('on_fly_coach.html',  # Changed from dashboard.html to index.html
#                                plot_html=plot_html,
#                                existing_files=get_existing_files(),
#                                current_filename=filename,
#                                dashboard_url=url_for('dashboard'))

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(url_for('dashboard'))
######################################

# Helper function to create health plots
def create_dataframe_from_form_data(data):
    """
    Create a pandas DataFrame from form data with all required columns
    """

    # Define all required columns for the DataFrame
    required_columns = [
        'Date', 'Player', 'Sleep quality', 'How many hours did you sleep?',
        'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level',
        'Calmness', 'Anger', 'Muscle Soreness', 'Comments', 'SalivaCoding',
        'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl',
        'Traces Blood Saliva', 'Ketones ng/ml', 'Cortisol ng/ml',
        'Cortisol nmol/L', 'Testosterone pg/ml', 'Testosterone nmol/L',
        'TCR', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score',
        'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score',
        'Biomarker_Factor', 'Total_Score', 'AST U/L', 'CK U/L',
        'DHEA-S ng/L', 'DHEA-S nmol/L', 'Gloominess'
    ]
    # Map form data to DataFrame columns
    # Convert string values to appropriate numeric types, defaulting to 0 if None or invalid
    def safe_convert(value, convert_type=float):
        if value is None or value == '':
            return 0
        try:
            return convert_type(value)
        except (ValueError, TypeError):
            return 0

    # Create the mapped data dictionary
    mapped_data = {
        'Date': datetime.now().strftime('%Y-%m-%d'),  # Current date
        'PlayeName': data.get('Name'),
        'Player':1,
        'Sleep quality': safe_convert(data.get('SleepQuality')),
        'How many hours did you sleep?': safe_float(data.get('SleepTime')),  # Not provided in form different in csv files fr atheletes
        'Sleep Time Score': safe_float(data.get('SleepTime')),  # Not provided in form
        'TrainingHours': safe_float(data.get('TrainingHours')),
        'ResistanceTraining': safe_float(data.get('ResistanceTraining')),
        'Tiredness': 1.0 - safe_convert(data.get('TirednessLevel')),
        'Energy level': safe_convert(data.get('EnergyLevel')),
        'Stress level': 1.0 - safe_convert(data.get('StressLevel')),
        'Calmness': safe_convert(data.get('CalmnessLevel')),
        'Gloominess': 1.0 - safe_convert(data.get('Gloominess')),
        'Muscle Soreness': 0,  # Not provided in form
        'Comments': '',  # Not provided in form
        'SalivaCoding': 0,  # Not provided in form
        'Salivary pH': safe_convert(data.get('pH')),
        'Salivary Nitrates': safe_convert(data.get('Nitrite')),
        'Salivary Uric Acid mg/dl': safe_convert(data.get('UricAcid')),
        'Traces Blood Saliva': 0,  # Not provided in form
        'Ketones ng/ml': safe_convert(data.get('Ketone')), # Not provided in form
        'Cortisol ng/ml': 0,  # Not provided in form
        'Cortisol nmol/L': 0,  # Not provided in form
        'Testosterone pg/ml': 0,  # Not provided in form
        'Testosterone nmol/L': 0,  # Not provided in form
        'TCR': 0,  # Not provided in form
        'Self_Reported_Score': 0,  # Not provided in form
        'pH_Score': safe_convert(data.get('pH')),  # Not provided in form
        'Nitrates_Score':safe_convert(data.get('Nitrite')),  # Not provided in form
        'Uric_Acid_Score':safe_convert(data.get('UricAcid')),  # Not provided in form
        'TCR_Score': 0,  # Not provided in form
        'Biomarker_Score': 0,  # Not provided in form
        'Biomarker_Factor': 0,  # Not provided in form
        'Total_Score': 0,  # Not provided in form
        'AST U/L': 0,  # Not provided in form
        'CK U/L': 0,  # Not provided in form
        'DHEA-S ng/L': 0,  # Not provided in form
        'DHEA-S nmol/L': 0  # Not provided in form
    }

    # Create DataFrame
    df = pd.DataFrame([mapped_data])

    return df

def process_dataframe_calculations(df):
    """
    Process the DataFrame with additional calculations and normalizations
    """
    # Create a copy to avoid modifying original
    data = df.copy()



    # Create reference row (you might want to modify this based on your reference values)
    reference_row = data.copy()
    reference_row['SQT'] = reference_row['Sleep Time Score'] * reference_row['Sleep quality']
    reference_row['TTQ'] = reference_row['TrainingHours'] * reference_row['ResistanceTraining']
    # Columns for radar chart normalization
    cols_to_normalize = ['STQ', 'TTQ', 'Tiredness', 'Energy level', 'Stress level', 'Calmness', 'Gloominess']

    # Normalize values (0-100 scale?  seems like 0 to 10)
    for col in cols_to_normalize:
        if col in data.columns:
            data[col] = (data[col] / 10) * 100
            reference_row[col] = (reference_row[col] / 10) * 100

    return data, reference_row, cols_to_normalize

def bio_total_score(ph, nitrite, uric_acid, selfscore):
    """
    Calculate ph, nitrite, uric acid scores and compute bioscore


    Returns:
       bioscore
    """
    # --- pH Score ---
    if ph == 7:
        ph_score = 100
    elif ph in [6, 8]:
        ph_score = 50
    else:
        ph_score = 0

    # --- Nitrite Score ---
    if nitrite == 1:
        nitrite_score = 50
    elif nitrite == 2:
        nitrite_score = 75
    elif nitrite == 3:
        nitrite_score = 100
    else:
        nitrite_score = 0

    # --- Uric Acid Score ---
    if uric_acid == 0:
        uric_acid_score = 0
    elif uric_acid == 0.5:
        uric_acid_score = 1
    elif 0.5 < uric_acid <= 2.5:
        uric_acid_score = 50
    elif 2.5 < uric_acid <= 5:
        uric_acid_score = 100
    else:
        uric_acid_score = 50

    # --- Calculate bioscore
    bioscore = (ph_score + nitrite_score + uric_acid_score) / 3
    if ph_score == 0 or nitrite_score == 0 or uric_acid_score == 0 or bioscore < 35:
        totalscore =  0
    else:
        totalscore =(bioscore / 100) * selfscore * 1.2

    return round(bioscore, 2), round(totalscore,2)



# Add this new route for creating and displaying plots
#make plot and use score card script

# Add route to view plots
@app.route('/view-plots')
@login_required
def view_plots():
    return render_template('view_plots.html')

# Add route to serve plot images
@app.route('/plot-image')
@login_required
def plot_image():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(base_dir, 'static', 'plots', 'health_assessment.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        flash("Plot not found. Please submit the form first.", "error")
        return redirect(url_for('health_assessment'))



##################################
@app.route('/health-assessment', methods=['GET'])
@login_required
def health_assessment():
    return render_template('VitalizeDx_questionnaire_template.html')

@app.route('/submit-health', methods=['POST'])
@login_required
def submit_health():
    try:
        data = {
        'Name': request.form.get('firstname'),
        'Surname': request.form.get('surname'),
        'Date': request.form.get('date'),
        'UID Number': request.form.get('uid'),
        'DOB': request.form.get('dob'),
        'Email': request.form.get('email'),
        'Gender': request.form.get('gender'),
        'Weight': request.form.get('weight'),
        'Height': request.form.get('height'),
        'BMI': request.form.get('bmi'),
        'Gloominess': request.form.get('mood'),
        'Diet': request.form.get('diet'),
        'TrainingHours': request.form.get('training_hour'),
        'ResistanceTraining': request.form.get('resistance'),
        'SleepQuality': request.form.get('sleep_quality'),      # NEW
        'SleepTime': request.form.get('sleep_time'),      # NEW
        'TirednessLevel': request.form.get('tiredness_level'),        # NEW
        'EnergyLevel': request.form.get('energy_level'),        # NEW
        'StressLevel': request.form.get('stress_level'),        # NEW
        'CalmnessLevel': request.form.get('calmness'),          # NEW
        'pH': request.form.get('ph'),
        'Nitrite': request.form.get('nitrite'),
        'UricAcid': request.form.get('uric_acid') #, 'Ketone': request.form.get('ketone')
        }

        # Validate required fields
        required_fields = ['firstname', 'surname', 'email']
        for field in required_fields:
            if not request.form.get(field):
                flash(f"Missing required field: {field}", "error")
                return redirect(url_for('health_assessment'))

        player_name = f"{data.get('Name', '').strip()} {data.get('Surname', '').strip()}"
        data_file =  os.path.join(app.config['PLOT_FOLDER'], 'online_data_submissions.csv')
        # Ensure data folder exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(base_dir, 'data')
        os.makedirs(data_folder, exist_ok=True)

#        data_file = os.path.join(data_folder, 'online_data_submissions.csv')

        with open(data_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        #reference and player data for Radar Plot
        # Extract values from data dictionary
        SleepQuality = safe_float(data.get('SleepQuality'))
        SleepTime = safe_float(data.get('SleepTime'))
        Training = safe_float(data.get('TrainingHours'))
        ResistanceTrain = safe_float(data.get('ResistanceTraining'))
        TirednessLevel = safe_float(data.get('TirednessLevel'))
        EnergyLevel = safe_float(data.get('EnergyLevel'))
        StressLevel = safe_float(data.get('StressLevel'))
        GlominessLevel =  safe_float(data.get('Gloominess'))
        CalmnessLevel = safe_float(data.get('CalmnessLevel'))
        # diet
        DietString = data.get('Diet') #balanced etc

        # Calculate metrics
        SQT = SleepQuality * SleepTime * 100
        TTQ = Training * ResistanceTrain * 100
        Tiredness = (1 -TirednessLevel)*100
        Energy = EnergyLevel*100
        Stress = (1 - StressLevel)*100# assuming it's in range [0, 1]
        Gloomy = (1 - GlominessLevel)*100
        Calmness = CalmnessLevel*100
        self_report_score = round((SQT + TTQ + Energy + Calmness + (100 -  Tiredness) + (100 - Stress)  + (100 - Gloomy))/7.0, 2)
        average_score = 75

        # Labels and values for radar
        #Radar categories
        categories = ['SQT', 'TTQ', 'Tiredness', 'Energy', 'Stress', 'Calmness', 'Gloomy']

        # Values for user and reference
        user_values = [SQT, TTQ, Tiredness, Energy, Stress, Calmness, Gloomy]
        reference_values = [45, 100, 25, 90, 40, 80, 25]
        # Close the radar loops
        categories += [categories[0]]
        user_values += [user_values[0]]
        reference_values += [reference_values[0]]


        fig = go.Figure()

        # User plot
        fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name=f"{data.get('Name', 'User')}'s Radar",
        line=dict(color='blue'),
        opacity=0.7
           ))

        # Reference plot
        fig.add_trace(go.Scatterpolar(
        r=reference_values,
        theta=categories,
        fill='toself',
        name="Reference Profile",
        line=dict(color='green', dash='dash'),
        opacity=0.5
            ))

        # Layout
        fig.update_layout(
        title="Wellness Radar Comparison",
        polar=dict(
        bgcolor='white',  # white background
        radialaxis=dict(
            visible=True,
            range=[0, max(reference_values + user_values) * 1.1],
            gridcolor='lightgrey',
            linecolor='black',
            tickfont=dict(size=10)  # smaller tick font for mobile
        ),
        angularaxis=dict(
            gridcolor='lightgrey',
            linecolor='black',
            tickfont=dict(size=10)  # smaller font for labels around the circle
        )
        ),
        showlegend=True,
        legend=dict(
        orientation="h",  # horizontal legend (better for narrow screens)
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=10)
        ),
        margin=dict(l=20, r=20, t=60, b=60),
        paper_bgcolor='white'
        )

        radar_plot1 = pio.to_html(fig, full_html=False)

        #Biomarker code from input values and total score bases on lavinia script add here and maybe make a radar and gaussion plot for that it maybe too much
        PHval = safe_float(data.get('pH'))
        Nitritval = safe_float(data.get('Nitrite'))
        uricacidval = safe_float(data.get('UricAcid'))
        #ketoneval = safe_float(data.get('Ketone'))
        biomark_score, combine_score  = bio_total_score(PHval, Nitritval, uricacidval, self_report_score)


        #create a panda framwork for score card aactions and other stuff
        df = create_dataframe_from_form_data(data)

        # Process calculations
        data, reference_row, cols_to_normalize = process_dataframe_calculations(df)
        #alternate I fix it now with dict
        player_values_dict = {
            'Sleep quality': SleepQuality *10,
            'Sleep Time Score': SleepTime *10,
            'Tiredness': (1 -TirednessLevel) *10,
            'Energy level':EnergyLevel*10,
            'Stress level': (1 - StressLevel)*10, # assuming it's in range [0, 1]
            'Gloomy' : (1 - GlominessLevel)*10,
            'Calmness':CalmnessLevel*10,
            'Anger': 0,
            'Muscle Soreness':0,
            'Salivary pH': PHval,
            'Salivary Uric Acid mg/dl':uricacidval,
            'Salivary Nitrates':Nitritval,
            #'Ketones ng/ml':0, #ketoneval,
            'Traces Blood Saliva': 0,
            'Cortisol nmol/L':0,
            'Testosterone nmol/L':0,
            'TCR':0
                }

        # Set player_id (you might want to get this from session or generate it)
        player_id = 1 #session.get('user_id', 1)  # or however you determine player ID
        timeline_plot = gaussianplot(self_report_score)


        #deviation_date_1 = (data_date_1[numeric_columns])# - player_averages[numeric_columns]) / player_averages[numeric_columns]
        benchmark_data = {
        'Data': [
            'Sleep quality', 'How many hours did you sleep?', 'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level',
            'Calmness', 'Anger', 'Muscle Soreness', 'SalivaCoding', 'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl',
            'Traces Blood Saliva', 'Ketones ng/ml', 'Cortisol ng/ml', 'Cortisol nmol/L', 'Testosterone pg/ml', 'Testosterone nmol/L',
           'TCR', 'Self_Reported_Score', 'pH_Score', 'Nitrates_Score', 'Uric_Acid_Score', 'TCR_Score', 'Biomarker_Score',
            'Biomarker_Factor', 'Total_Score', 'AST U/L', 'CK U/L', 'DHEA-S ng/L', 'DHEA-S nmol/L'
        ],
        'Benchmark Mean': [
            8, None, 1, 1, 8, 1, 10, 1, 1, 2025012913, 7, 2, 5, 0.5, 1.5, 8, 22.08, 300, 1.041, 0.047, 82.5, 100, 100, 100, 100, 100, 1, 99, 30, 50, 5, 13.55
        ],
        'Benchmark Min': [
            7, None, 0.8, 0.8, 6, 0.8, 8, 0.8, 0.8, 2025012913, 6, 1, 2, 0, 0, 4, 14, 150, 0, 0, 64, 50, 50, 50, 50, 50, 0.5, 49.5, 15, 25, 2.5, 6.775
        ],
        'Benchmark Max': [
            9, None, 1.2, 1.2, 10, 1.2, 12, 1.2, 1.2, 2025012913, 8, 3, 7, 1, 3, 12, 30, 450, 2, 0.11, 100, 150, 150, 150, 150, 150, 1.5, 150, 58.5, 97.5, 9.75, 26.4225
        ]
        }
        benchmark_df = pd.DataFrame(benchmark_data)


        current_date = datetime.now().strftime('%m/%d/%Y')

        diet_dict = {player_id: DietString}  #for new actions in  Ketone
        result = evaluate_single_player_data_v2(player_values_dict, player_id, current_date, diet_dict)

        ##### wrong way
        results_1 = evaluate_single_player_data(player_values_dict, player_id, current_date, diet_dict)

        df = results_1[['Player', 'Parameter', 'Player Value', 'Deviation from Benchmark', 'Deviation from Average','Date', 'Direct Associations', 'Actions']]
        df = df.dropna(subset=['Player', 'Parameter', 'Player Value'])

        specified_metrics = ['Sleep quality', 'Sleep Time Score', 'Tiredness', 'Energy level', 'Stress level', 'Calmness', 'Gloominess',
            'Salivary pH', 'Salivary Nitrates', 'Salivary Uric Acid mg/dl', #'Ketones ng/ml',
            'Cortisol nmol/L', 'Testosterone nmol/L', 'Biomarker_Score', 'Total_Score', 'Self_Reported_Score'
            ]
        df = df[df['Parameter'].isin(specified_metrics)]




        metrics_html = ""
        total_score = 0
        biomarker_score = 0
        self_reported_score = 0
        total_score_change = 0
        biomarker_score_change = 0
        self_reported_score_change = 0
        # Retrieve all actions for the player

#commenting below for new
        actions_row = df[df['Actions'].notna() & (df['Actions'].str.strip() != "")]
        if not actions_row.empty:
            # Get all actions as a single string joined by slashes
            all_actions = " / ".join(actions_row['Actions'].dropna().astype(str).unique())

            # Split the string by slashes to create individual action items
            action_items = [item.strip() for item in all_actions.split('/') if item.strip()]

            # Create HTML bullet list with center alignment
            if action_items:
                actions_html = "<ul style='list-style-type: disc; margin: 0; padding-left: 20px;'>"
                #actions_html = "<ul style='list-style-type: disc; margin: 0 auto; padding: 0; text-align: justify; list-style-position: inside;'>"
                for item in action_items:
                    actions_html += f"<li style='margin-bottom: 5px; text-align: left;'>{item}</li>"
                    #actions_html += f"<li style='display: block; margin-bottom: 5px; text-align: justify;'>{item}</li>"
                actions_html += "</ul>"
                actions = actions_html
            else:
                actions = "<div style='text-align: justify;'>No actions specified</div>"
        else:
            actions = "<div style='text-align: justify;'>No actions specified</div>"

        # Extract actions
#        action_items = result.get("Actions", [])

#        if action_items:
#            actions_html = "<ul style='list-style-type: disc; margin: 0 auto; padding: 0; text-align: justify; list-style-position: inside;'>"
#            for item in action_items:
#                actions_html += f"<li style='display: block; margin-bottom: 5px; text-align: justify;'>{item}</li>"
#            actions_html += "</ul>"
#            actions = actions_html
#        else:
#            actions = "<div style='text-align: justify;'>No actions specified</div>"

        #Handle self-reported scores for this player
        self_reported_rows = df[df['Parameter'] == 'Self_Reported_Score']
        if not self_reported_rows.empty:
            latest_date = self_reported_rows['Date'].max()
            latest_score_row = self_reported_rows[self_reported_rows['Date'] == latest_date]

            if not latest_score_row.empty:
                self_reported_score = round(latest_score_row['Player Value'].values[0], 2)
                self_reported_score_change = round(latest_score_row['Deviation from Average'].values[0] * 100, 2)

        # Process metrics for this player
        for index, row in df.iterrows():
            if row['Parameter'] == 'Total_Score' and pd.notna(row['Player Value']):
                total_score = round(row['Player Value'], 2)
                total_score_change = round(row['Deviation from Average'] * 100, 2) if pd.notna(row['Deviation from Average']) else 0
            elif row['Parameter'] == 'Biomarker_Score' and pd.notna(row['Player Value']):
                biomarker_score = round(row['Player Value'], 2)
                biomarker_score_change = round(row['Deviation from Average'] * 100, 2) if pd.notna(row['Deviation from Average']) else 0

        # Determine the color for the scores
        total_score_color = "#4caf50" if total_score > 50 else "#f44336"
        biomarker_score_color = "#4caf50" if biomarker_score > 50 else "#f44336"
        self_reported_score_color = "#4caf50" if self_reported_score > 50 else "#f44336"

        # Determine the change class and icon for scores
        # Determine the change class and icon for scores, with empty icons
        total_score_change_class = "up" if total_score_change > 0 else "down"
        total_score_change_icon = ""  # No arrow

        biomarker_score_change_class = "up" if biomarker_score_change > 0 else "down"
        biomarker_score_change_icon = ""  # No arrow

        self_reported_score_change_class = "up" if self_reported_score_change > 0 else "down"
        self_reported_score_change_icon = ""  # No arrow

        #total_score_change_class = "up" if total_score_change > 0 else "down"
        #total_score_change_icon = "🔼" if total_score_change > 0 else "🔽"
        #biomarker_score_change_class = "up" if biomarker_score_change > 0 else "down"
        #biomarker_score_change_icon = "🔼" if biomarker_score_change > 0 else "🔽"
        #self_reported_score_change_class = "up" if self_reported_score_change > 0 else "down"
        #self_reported_score_change_icon = "🔼" if self_reported_score_change > 0 else "🔽"

        # Create organized metrics for left and right columns
        left_metrics_html = ""
        right_metrics_html = ""

        # Define the order for left and right columns
        left_metrics_order = ['Salivary pH', 'Salivary Nitrates']#,
        right_metrics_order = ['Salivary Uric Acid mg/dl', 'Ketones ng/ml']
        #right_metrics_order = ['Cortisol nmol/L', 'Testosterone nmol/L', 'TCR']

        # Process metrics for left column (pH, Nitrates, Uric Acid, Ketones)
        for metric_name in left_metrics_order:
            metric_rows = df[df['Parameter'] == metric_name]
            if not metric_rows.empty:
                row = metric_rows.iloc[0]  # Take the first row for this metric
                if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
                    benchmark_row = benchmark_df[benchmark_df['Data'] == metric_name]
                    if not benchmark_row.empty:
                        benchmark_min = benchmark_row['Benchmark Min'].values[0]
                        benchmark_max = benchmark_row['Benchmark Max'].values[0]
                        benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
                    else:
                        print(f"WARNING: No benchmark found for {metric_name}")
                        benchmark_min = None
                        benchmark_max = None
                        benchmark_mean = None

                    direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

                    left_metrics_html += generate_metric_html(
                    metric_name, row['Player Value'], row['Deviation from Average'],
                    benchmark_min, benchmark_max, benchmark_mean, direct_association
                    )

        # Process metrics for right column (Cortisol, Testosterone)
        for metric_name in right_metrics_order:# [:2]:  # Only process Cortisol and Testosterone here
            metric_rows = df[df['Parameter'] == metric_name]
            if not metric_rows.empty:
                row = metric_rows.iloc[0]  # Take the first row for this metric
                if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
                    benchmark_row = benchmark_df[benchmark_df['Data'] == metric_name]
                    if not benchmark_row.empty:
                        benchmark_min = benchmark_row['Benchmark Min'].values[0]
                        benchmark_max = benchmark_row['Benchmark Max'].values[0]
                        benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
                else:
                    print(f"WARNING: No benchmark found for {metric_name}")
                    benchmark_min = None
                    benchmark_max = None
                    benchmark_mean = None

                direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

                right_metrics_html += generate_metric_html(
                    metric_name, row['Player Value'], row['Deviation from Average'],
                    benchmark_min, benchmark_max, benchmark_mean, direct_association
                )

        # Process TCR separately
        #tcr_rows = df[df['Parameter'] == 'TCR']
        #if not tcr_rows.empty:
        #    row = tcr_rows.iloc[0]
        #    if pd.notna(row['Player Value']) and pd.notna(row['Deviation from Average']):
        #        benchmark_row = benchmark_df[benchmark_df['Data'] == 'TCR']
        #        if not benchmark_row.empty:
        #            benchmark_min = benchmark_row['Benchmark Min'].values[0]
        #            benchmark_max = benchmark_row['Benchmark Max'].values[0]
        #            benchmark_mean = benchmark_row['Benchmark Mean'].values[0]
        #        else:
        #            print(f"WARNING: No benchmark found for TCR")
        #            benchmark_min = None
        #            benchmark_max = None
        #            benchmark_mean = None

         #       direct_association = row['Direct Associations'] if pd.notna(row['Direct Associations']) else ""

                # Add TCR with actions at the bottom of the right column
         #       right_metrics_html += generate_tcr_metric_html(
         #           row['Player Value'], row['Deviation from Average'],
         #           benchmark_min, benchmark_max, benchmark_mean, direct_association, actions
         #       )
        #else:
        #    print(f"WARNING: No TCR data found for Player {int(player_id)}")


        return render_template('visitor_repoty.html',
                        player_id=player_id,
                        player_name = player_name,
                        radar_plot=radar_plot1,
                        timeline_plot=timeline_plot, #fixc it
                        actions =actions,
                        total_score= combine_score,
                        total_score_change=abs(total_score_change),  # Use absolute value for display
                        total_score_change_class=total_score_change_class,
                        total_score_change_icon=total_score_change_icon,
                        biomarker_score=biomark_score,
                        biomarker_score_change=abs(biomarker_score_change),  # Use absolute value for display
                        biomarker_score_change_class=biomarker_score_change_class,
                        biomarker_score_change_icon=biomarker_score_change_icon,
                        self_reported_score=self_report_score,
                        self_reported_score_change=abs(self_reported_score_change),  # Use absolute value for display
                        self_reported_score_change_class=self_reported_score_change_class,
                        self_reported_score_change_icon=self_reported_score_change_icon,
                        total_score_color=total_score_color,
                        biomarker_score_color=biomarker_score_color,
                        self_reported_score_color=self_reported_score_color,
                        left_metrics_html=left_metrics_html,
                        right_metrics_html=right_metrics_html,
                        current_year=datetime.now().year)#
                         #scores_html=scores_html,
                         #metrics_html=metrics_html)

        #output_file_path = r"Player_Evaluation_Results_With_Deviation_And_Date.csv"
        #combined_results.to_csv(output_file_path, index=False)



#        flash("Health data submitted successfully!", "success")
#        return redirect(url_for('health_assessment'))

    except Exception as e:
        flash(f"Error saving data: {e}", "error")
        return redirect(url_for('health_assessment'))



if __name__ == '__main__':
    app.run(debug=True)
