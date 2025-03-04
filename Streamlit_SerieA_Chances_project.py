import os
import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.stats import binned_statistic_2d
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon, Arrow, ArrowStyle,FancyArrowPatch, Circle,FancyArrow
from mplsoccer.pitch import Pitch, VerticalPitch
from matplotlib.colors import Normalize
from matplotlib import cm
from highlight_text import fig_text, ax_text

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings("ignore")

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import matplotlib.image as mpimg
import matplotlib.patches as patches
from io import BytesIO
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.markers import MarkerStyle
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbopen, add_image
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.patheffects import withStroke, Normal
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer.utils import FontManager
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.cluster import KMeans
import warnings
from highlight_text import ax_text, fig_text
from PIL import Image
from urllib.request import urlopen
import os
import time
from unidecode import unidecode
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors
import streamlit as st
URL = 'https://github.com/googlefonts/BevanFont/blob/main/fonts/ttf/Bevan-Regular.ttf?raw=true'
fprop = FontManager(URL).prop
img_save_loc='D:\CoCoding\DJ\Football Analytics\Data Visualization\\MatchReports'

st.title("세리에 A 24 25 기회 창출")
st.subheader("세트피스를 제외한 플레이만 표시합니다.")

# Load Data
# Google Drive Direct Link to ZIP (Replace FILE_ID with your actual ID)

df = pd.read_csv('SerieA2425.csv')
# Column mapping
column_mapping = {
    'match_id':'matchId',
    'event_id': 'eventId',
    'expanded_minute': 'expandedMinute',
    'minute': 'minute',
    'second': 'second',
    'team_id': 'teamId',
    'player_id': 'playerId',
    'x': 'x',
    'y': 'y',
    'end_x': 'endX',
    'end_y': 'endY',
    'satisfied_events_types': 'satisfiedEventsTypes',
    'is_touch': 'isTouch',
    'blocked_x': 'blockedX',
    'blocked_y': 'blockedY',
    'goal_mouth_z': 'goalMouthZ',
    'goal_mouth_y': 'goalMouthY',
    'related_event_id': 'relatedEventId',
    'related_player_id': 'relatedPlayerId',
    'is_shot': 'isShot',
    'card_type': 'cardType',
    'is_goal': 'isGoal',
    'outcome_type': 'outcomeType',
    'period_display_name': 'period',
    'shirt_no': 'shirtNo',
    'is_first_eleven': 'isFirstEleven'
}
df = df.rename(columns=column_mapping)

# Define the teamId mapping dictionary
team_id_mapping = {
    'Atalanta': 300, 'Bologna': 71, 'Cagliari': 78, 'Como': 1290, 'Empoli': 272,
    'Fiorentina': 73, 'Genoa': 278, 'Verona': 76, 'Inter': 75, 'Juventus': 87,
    'Lazio': 77, 'Lecce': 79, 'AC Milan': 80, 'Monza': 269, 'Napoli': 276,
    'Parma': 24341, 'AS Roma': 84, 'Torino': 72, 'Udinese': 86, 'Venezia': 85
}

# Reverse the dictionary to map teamId to teamName
team_id_to_name = {v: k for k, v in team_id_mapping.items()}

# Add the new column "teamName" using the mapping
df["teamName"] = df["teamId"].map(team_id_to_name)

ftmb_tid_mapping = {
    'Inter': 8636,
    'Genoa': 10233,
    'Parma': 10167,
    'Udinese': 8600,
    'Verona': 9876,
    'Juventus': 9885,
    'Lazio': 8543,
    'Torino': 9804,
    'Bologna': 9857,
    'Fiorentina': 8535,
    'Cagliari': 8529,
    'Empoli': 8534,
    'AS Roma': 8686,
    'AC Milan': 8564,
    'Monza': 6504,
    'Venezia': 7881,
    'Como': 10171,
    'Napoli': 9875,
    'Atalanta':8524,
    'Lecce':9888
}
df['ftmb_tid'] = df['teamName'].map(ftmb_tid_mapping)
df['ftmb_tid'] = df['ftmb_tid'].fillna(0).astype(int)

# Remove set-piece passes
df = df[~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn', na=False)]
df=df[df['type']=='Pass']

# Select Team & Player
team = st.selectbox('팀을 선택하세요', df['teamName'].sort_values().unique(), index=None)
player = st.selectbox('Select a player', df[df['teamName'] == team]['name'].sort_values().unique(), index=None)

# Function to filter data
def filter_data(df, team, name):
    if team:
        df = df[df['teamName'] == team]  
    if name:
        df = df[df['name'] == name]  
    return df  # Always return DataFrame (Never None)

filtered_df = filter_data(df, team, player)

# Ensure `filtered_df` is valid before plotting
if filtered_df is None or filtered_df.empty:
    st.warning("No data available for the selected player.")
    filtered_df = pd.DataFrame(columns=df.columns)  # Avoid errors

# Create Pitch
pitch = VerticalPitch(pitch_type='opta', pitch_color='#ffffff', line_color='#8D8D8B', line_zorder=0.1, linewidth=1, half=True, goal_type='box', pad_bottom=10)
fig, ax = pitch.draw(figsize=(10, 10))
ax.set_ylim(49.9, 100)


# Function to plot passes
def plot_chances_created(df, ax, pitch):
    if df is None or df.empty:
        return  # Prevents errors when no data is available
    
    df = df[df['qualifiers'].str.contains('KeyPass|BigChanceCreated|IntentionalGoalAssist', na=False)]
    
    for x in df.to_dict(orient='records'):


        pitch.lines(
        xstart=float(x['x']),
        ystart=float(x['y']),
        xend=float(x['endX']),
        yend=float(x['endY']),
        lw=3, comet=True, 
        color='#aa65b2' if 'IntentionalGoalAssist' in x['qualifiers'] else '#FD4890', 
        ax=ax, alpha=0.5  # Blue for Key Passes
    )
        pitch.scatter(
            x=float(x['endX']),  
            y=float(x['endY']),
            s=55,
            color='white',
            edgecolor='#aa65b2' if 'IntentionalGoalAssist' in x['qualifiers'] else '#FD4890',  
            linewidth=1, 
            zorder=2, 
            ax=ax
        )

# Plot Passes
plot_chances_created(filtered_df, ax, pitch)

head_length = 0.3
head_width = 0.07
# Annotate Pitch

ax.annotate(xy=(104, 60),zorder=2,
                   text='Attack',
                   ha='center',
                   color='#000000',
                   rotation=90,
                   fontproperties=fprop,fontsize=13)
ax.annotate(xy=(102, 70), 
                xytext=(102, 55),zorder=2,
                text='',
                ha='center',
                arrowprops=dict(arrowstyle=f'->, head_length = {head_length}, head_width={head_width}',
                color='#000000',
                lw=0.5))

# Filter for assists made by the selected player
# Filter for each type of pass event
g_assist = df[(df['teamName'] == team) & (df['name'] == player) & (df['qualifiers'].str.contains('IntentionalGoalAssist', na=False))]
key_pass = df[(df['teamName'] == team) & (df['name'] == player) & (df['qualifiers'].str.contains('KeyPass', na=False))]
big_chance = df[(df['teamName'] == team) & (df['name'] == player) & (df['qualifiers'].str.contains('BigChanceCreated', na=False))]

# Find exact duplicate events between key_pass and big_chance
duplicates = key_pass.merge(big_chance, on=['matchId', 'eventId', 'x', 'y'], how='inner')

# Drop these duplicates from both key_pass and big_chance
key_pass = key_pass[~key_pass.set_index(['matchId', 'eventId', 'x', 'y']).index.isin(duplicates.set_index(['matchId', 'eventId', 'x', 'y']).index)]
big_chance = big_chance[~big_chance.set_index(['matchId', 'eventId', 'x', 'y']).index.isin(duplicates.set_index(['matchId', 'eventId', 'x', 'y']).index)]

# Get final counts
assist_count = len(g_assist)
chance_count = len(big_chance) + len(key_pass)
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load the uploaded font
font_path = "NanumGothic-Regular.ttf"  # Ensure this file exists in your project folder
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()  # Set the custom font
import matplotlib.patches as patches

# Add text for assists
fig.text(0.19, 0.88, f"도움 ({assist_count})", fontsize=15, ha='left', va='center', color='#000000')

# Add a circle next to the text with adjustable size and edge color
circle_size = 0.01  # Adjust size of the circle
circle = patches.Circle((0.28, 0.88), circle_size, transform=fig.transFigure, 
                        facecolor='white', edgecolor='#aa65b2', linewidth=1.5, zorder=3)
circle2 = patches.Circle((0.45, 0.88), circle_size, transform=fig.transFigure, 
                        facecolor='white', edgecolor='#FD4890', linewidth=1.5, zorder=3)
# Add the circle to the figure
fig.patches.append(circle)  # Add circle to figure
fig.patches.append(circle2)  # Add circle to figure
fig.text(0.19, 0.88, f"도움 ({assist_count})", fontsize=15,
         ha='left', va='center', color='#000000')
fig.text(0.3, 0.88, f"기회 창출 ({chance_count})", fontsize=15,
         ha='left', va='center', color='#000000')
fig.text(0.18, 0.95, f"{player}", fontsize=30, fontweight='bold', ha='left', va='center', fontproperties=fprop, color='#000000')
fig.text(0.19, 0.91, f'세리에 A 2024-25 | 1라운드부터 26까지', fontsize=15, ha='left', va='center')

# Get ftmb_tid for the selected team (make sure it's an integer)
ftmb_tid = df.loc[df['teamName'] == team, 'ftmb_tid'].values
ftmb_tid = int(ftmb_tid[0]) if len(ftmb_tid) > 0 else None

if ftmb_tid is not None:
    # Download the image using urlopen
    himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{ftmb_tid}.png")
    himage = Image.open(himage)

    # Add the image to the figure at a custom position
    ax_himage = add_image(himage, fig, left=0.04, bottom=0.85
    , width=0.125, height=0.125)

# Plot the line
ax.plot([0.6, 0.85], [0.91, 0.91],  # [x_start, x_end], [y_start, y_end]
        color='black', lw=1, transform=fig.transFigure)

# Plot scatter points at start and end of the line
ax.scatter([0.6, 0.85], [0.91, 0.91],  # [x_positions], [y_positions]
           color='red', edgecolor='black', s=50, transform=fig.transFigure)
# Show Plot
circle3 = patches.Circle((0.8, 0.88), circle_size, transform=fig.transFigure, 
                        facecolor='white', edgecolor='#FD4890', linewidth=1.5, zorder=3)
fig.patches.append(circle3)  # Add circle to figure

# Define the line position (slightly to the left of the circle)
line_x_start, line_x_end = 0.68, 0.8  # Adjust X values to control position
line_y_start, line_y_end = 0.88, 0.88  # Keep Y values the same to make a horizontal line

# Add the line to the figure
line = ax.plot([line_x_start, line_x_end], [line_y_start, line_y_end],  
               color='#FD4890', lw=3, transform=fig.transFigure, clip_on=False)

fig.canvas.draw()

fig.text(0.56, 0.88, f"패스한 위치", fontsize=15,
         ha='left', va='center', color='#000000')
fig.text(0.82, 0.88, f"패스 받은 위치", fontsize=15,
         ha='left', va='center', color='#000000')
plt.show()
st.pyplot(fig)
