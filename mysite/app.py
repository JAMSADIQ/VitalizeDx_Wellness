import os
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import sys

path = '/home/JamSadiq/mysite'
if path not in sys.path:
    sys.path.append(path)

app = Flask(__name__)
app.secret_key = 'd2c8f1e7b5a9d4f6c3e8b2a7d5f1e3c'  # Change this to a strong random key in production


# Session security
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# Configuration
UPLOAD_FOLDER = '/home/JamSadiq/mysite/uploads'
PLOT_FOLDER = '/home/JamSadiq/mysite/static'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User configuration (replace with database in production)
users = {
    'admin': {
        'password': 'mypass',  # Make sure this is exactly what you're typing
        'name': 'admin'
    },
    'user': {
        'password': 'userpass',  # Make sure this is exactly what you're typing
        'name': 'user'
    }
}

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    pass



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

    loss_dates = data[(data['Match_Outcome'] == 2) & (data['Date'] <= cutoff)]['Date'].unique()
    return sorted(loss_dates)
### I need to add one that find the exclusion dates from file
############################################################


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
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


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
        # Store the responses
        filename = request.form.get('filename')
        responses = {
            'q1': int(request.form.get('q1')),
            'q2': int(request.form.get('q2')),
            'q3': int(request.form.get('q3')),
            'q4': int(request.form.get('q4')),
            'q5': int(request.form.get('q5')),
            'q6': int(request.form.get('q6')),
            'q7': int(request.form.get('q7')),
            'comments': request.form.get('comments', '')
        }
        questionnaire_responses[filename] = responses
        flash('Questionnaire submitted successfully!', 'success')
        # Redirect to first page in the sequence
        return redirect(url_for('show_sequence', page_index=0))

    filename = request.args.get('filename')
    return render_template('questionnaire.html', filename=filename)

@app.route('/show_sequence/<int:page_index>')
@login_required
def show_sequence(page_index):
    html_pages = [
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



@app.route('/plot', methods=['POST'])
@login_required
def plot():
    selected_date = None
#    # Check if using existing file
    if 'existing_file' in request.form and request.form['existing_file']:
        filename = request.form['existing_file']
#    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'SummaryTesting_Numeric_WithScores_Calculated.csv')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Otherwise handle file upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('dashboard'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            flash('Allowed file types are .csv and .xlsx', 'error')
            return redirect(url_for('dashboard'))
    else:
        flash('No file provided', 'error')
        return redirect(url_for('dashboard'))

#    # Special handling for data_sin_plot.xlsx
#    if filename == 'data_sin_plot.xlsx':
#        try:
#            df = pd.read_excel(filepath)
#            if len(df.columns) < 2:
#                flash('Excel file must have at least two columns', 'error')
#                return redirect(url_for('dashboard'))
#
#            fig = px.line(df, x=df.columns[0], y=df.columns[1], title='Sine Wave Plot')
#            plot_html = fig.to_html(full_html=False)
#
#            return render_template('dashboard.html',
#                                plot_html=plot_html,
#                                existing_files=get_existing_files(),
#                                current_filename=filename)

#        except Exception as e:
#            flash(f'Error processing Excel file: {str(e)}', 'error')
#            return redirect(url_for('dashboard'))

#    # Original CSV processing for other files
    try:
        #belowuncomment if we can run the code and update by any user
#        script_path = os.path.join(app.static_folder, 'plot_classify.py')
#        subprocess.run(['python', script_path, filepath], check=True)
#        base_name = os.path.splitext(filename)[0]
#        output_dir = os.path.join('static', 'plots', base_name)


        # Load the CSV file
        data = pd.read_csv(filepath)

        # Rename columns for consistency
        data.rename(columns={
            'Sleep Time Score_x': 'Sleep Time Score',
            'Performance_x': 'Performance',
            'Date_x': 'Date'
        }, inplace=True)

        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        #Here I want to fix: So chnage dashboard so hte drop down menu show available dates and what we choose is selected date
        available_dates = sorted(data['Date'].dt.strftime('%d/%m/%Y').unique())  # Format as DD/MM/YYYY for display

        if request.method == 'POST' and 'selected_date' in request.form:
            selected_date_str = request.form.get('selected_date')
            selected_date = pd.to_datetime(selected_date_str, format='%d/%m/%Y')  # Match the

        #cutoff must be taken from option
        dates_of_interest = get_dates_up_to_cutoff(data, cutoff_date=selected_date)
        loss_dates = get_match_loss_dates(data, selected_date)

        # Define the dates of interest
        #dates_of_interest = [
        #    "14/01/2025", "22/01/2025", "29/01/2025",
        #    "5/02/2025", "26/02/2025", "5/03/2025", "12/03/2025", "19/03/2025"
        #]
        #dates_of_interest = [pd.to_datetime(date, format='%d/%m/%Y') for date in dates_of_interest]

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
        11: ["3/19/2025", "3/26/2025", "4/2/2025", "4/9/2025"],
        1: ["3/26/2025", "4/9/2025", "4/16/2025"]  # Added exclusions for Player 1
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
        fig.update_traces( marker=dict(size=10), line=dict(width=3) )
        # Add vertical lines for match losses
        #loss_dates = [pd.to_datetime(date, format='%d/%m/%Y') for date in ["22/01/2025", "5/02/2025", "12/03/2025", "09/04/2025"]]
        for date in loss_dates:
            fig.add_vline(x=date, line_dash="dot", line_color="red", line_width=2, opacity=0.7)

        # Update layout to match your matplotlib style
        fig.update_layout( xaxis_title='Date',
        yaxis_title='Median Score',
        legend_title='',
        font=dict(
            size=18 ),
        title_font_size=24,
        title_font=dict(
            weight='bold'
        ),
        xaxis=dict(
            tickfont=dict(size=20),
            tickangle=-45
        ),
        yaxis=dict(
            tickfont=dict(size=20)
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0.9)'
            )
        )
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot')


        output_path = os.path.join(app.config['PLOT_FOLDER'], 'aggregateplot.png')
        fig.write_image(output_path)
        plot_html = fig.to_html(full_html=False)



        #### PCA plot
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
        principal_components = pca.fit_transform(scaled_data)#

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(principal_components)
        pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = clusters
        pca_df['Player'] = player_ids.values
        cluster_colors = ['#1a2a6c', '#b21f1f', '#1e988a']  # Example: blue, red, yellow
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

        # Add cluster labels at centroids
        for cluster_id in range(3):
            cluster_data = pca_df[pca_df['Cluster'] == cluster_id]
            centroid = cluster_data[['PC1', 'PC2']].mean()#

            fig.add_annotation(
                x=centroid['PC1'],
                y=centroid['PC2'],
                text=f"Cluster {cluster_id}",
                showarrow=False,
                font=dict(size=22, color='white', family="Arial Black"),
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor=cluster_colors[cluster_id],
                opacity=0.7
                )
        # Add player numbers as text
        for i, row in pca_df.iterrows():
            fig.add_annotation(
                x=row['PC1'],
                y=row['PC2'],
                text=str(int(row['Player'])),
                showarrow=False,
                font=dict(size=20, color='white', family="Arial Black")
                )

        # Update layout
        fig.update_layout(
            xaxis_title='Principal Component 1',
           yaxis_title='Principal Component 2',
            title_font=dict(size=24, family="Arial Black"),
            xaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
            yaxis=dict(title_font=dict(size=22), tickfont=dict(size=18)),
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


        model = RandomForestRegressor(random_state=42, max_depth=3)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        feature_names = X.columns
        feature_importance = np.abs(shap_values).mean(0)
        sorted_idx = np.argsort(feature_importance)

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
            hovertemplate='%{y}: %{x:.2f}<extra></extra>'
            ))

        # Add value labels
        for i, v in enumerate(sorted_values):
            fig.add_annotation(
                x=v + 0.01,
                y=i,
                text=f"{v:.2f}",
                showarrow=False,
                font=dict(size=20, color='black', family="Arial Black"),
                xanchor='left'
                )

        # Update layout
        fig.update_layout(
            title='Feature Importance (SHAP Values)',
            xaxis_title='mean(|SHAP value|)',
            margin=dict(l=150, r=50, b=50, t=100),
            title_font=dict(size=24, family="Arial Black"),
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
        return render_template('dashboard.html',
                            plot_html=plot_html,
                            plot_html2=plot_html2,
                            plot_html3=plot_html3,
                            existing_files=get_existing_files(),
                             selected_file=filename,  # Required for the second dropdown
                             available_dates=available_dates,
                             selected_date=selected_date,  # Pass to tem
                            current_filename=filename)


# uncomment below if update of user is working


#        return render_template('on_fly_coach.html',  # Changed from dashboard.html to index.html
#                                plot_html=plot_html,
#                                existing_files=get_existing_files(),
#                                current_filename=filename,
#                                dashboard_url=url_for('dashboard'))

    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))



def get_existing_files():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        return [f for f in os.listdir(app.config['UPLOAD_FOLDER'])
               if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]
    return []

if __name__ == '__main__':
    app.run(debug=True)
