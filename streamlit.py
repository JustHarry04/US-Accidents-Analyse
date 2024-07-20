import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import calendar
import scipy.stats as stats
import pylab
import folium 
import numpy as np
from folium.plugins import HeatMap


df = pd.read_csv('Us_accidents_Cleaned.csv')

st.sidebar.title("US-Accidents Analysis")
if st.sidebar.button("Analysis"):
    st.session_state.page = "analysis"
if st.sidebar.button("Hypothesis"):
    st.session_state.page = "Hypothesis"
if st.sidebar.button("Back to Overview"):
        st.session_state.page = "Overview"
        st.experimental_rerun()


if 'page' not in st.session_state:
    st.session_state.page = "Overview"

if st.session_state.page == "Overview":
    st.title("US-Accidents Analysis (2016-2022)")
    st.write("**Dataset Overview**")
    st.write(df.head())

    st.write("# The shape of the Dataset")
    st.write(df.shape)

    overview_text = """
    **Overview**

    The US-Accidents dataset is a comprehensive collection of traffic accident data across the United States, spanning from 2016 to 2022. This dataset provides valuable insights into various factors contributing to traffic accidents, including temporal, spatial, environmental, and other contextual data.

    **Data Collection**

    The data is compiled from several traffic authorities and sources, which are systematically recorded to reflect real-world incidents accurately. The dataset includes over 4.2 million accident records, making it one of the largest publicly available resources for traffic accident analysis.

    **Key Features**

    The dataset contains 47 columns, each representing a different attribute related to the accidents. Below are some of the key features:

    - **ID**: Unique identifier for each accident record.
    - **Source**: Source from which the accident data is obtained.
    - **Severity**: Severity of the accident on a scale of 1 to 4 (1 being the least severe and 4 being the most severe).
    - **Start_Time**: Start time of the accident.
    - **End_Time**: End time of the accident.
    - **Start_Lat**: Starting latitude of the accident.
    - **Start_Lng**: Starting longitude of the accident.
    - **Distance(mi)**: Distance over which the accident occurred.
    - **Description**: Text description of the accident.
    - **City**: City where the accident occurred.
    - **County**: County where the accident occurred.
    - **State**: State where the accident occurred.
    - **Zipcode**: Zip code of the accident location.
    - **Country**: Country where the accident occurred (only includes US).
    - **Timezone**: Timezone of the accident location.
    - **Airport_Code**: Nearest airport to the accident location.
    - **Weather_Timestamp**: Time when weather data was recorded.
    - **Temperature(F)**: Temperature in Fahrenheit at the time of the accident.
    - **Wind_Chill(F)**: Wind chill in Fahrenheit at the time of the accident.
    - **Humidity(%)**: Humidity percentage at the time of the accident.
    - **Pressure(in)**: Atmospheric pressure in inches at the time of the accident.
    - **Visibility(mi)**: Visibility in miles at the time of the accident.
    - **Wind_Direction**: Wind direction at the time of the accident.
    - **Wind_Speed(mph)**: Wind speed in miles per hour at the time of the accident.
    - **Precipitation(in)**: Precipitation in inches at the time of the accident.
    - **Weather_Condition**: Weather condition at the time of the accident.
    - **Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop**: Various indicators of road infrastructure and conditions present at the accident location.
    - **Sunrise_Sunset**: Indicates whether the accident occurred during day or night.
    - **Civil_Twilight, Nautical_Twilight, Astronomical_Twilight**: Twilight conditions at the time of the accident.

    The US-Accidents dataset provides a rich source of information for understanding and mitigating traffic accidents in the United States, offering numerous opportunities for in-depth analysis and practical applications.
    """

    st.markdown(overview_text)

elif st.session_state.page == "analysis":
    st.title("US-Accidents Dataset Analysis (2016-2020)")
    total_accidents = df.groupby('State').size().reset_index(name='Count')
    fig = px.choropleth(total_accidents,
                        locations='State',
                        locationmode='USA-states',
                        color='Count',
                        color_continuous_scale='Viridis',
                        scope='usa',
                        title='Total Accident Reports (2016 - 2020)')
    st.plotly_chart(fig)
    description = """
    ### Total Accident Reports (2016 - 2020)

    This choropleth map visualizes the total number of accident reports recorded in each state across the United States from 2016 to 2020. The color intensity on the map represents the accident count, with darker shades indicating higher numbers of accidents.

    **Key Insights:**
    - The states with the highest number of accident reports are highlighted in darker shades of color.
    - This visualization helps identify which states have higher incidences of traffic accidents, which can be crucial for traffic authorities and policymakers to focus their safety measures and resources.
    - States like California (CA), Texas (TX), and Florida (FL) are among those with the highest accident counts during this period.

    This map serves as an essential tool for understanding the geographical distribution of traffic accidents and can aid in targeted interventions to improve road safety.
    """
    st.markdown(description)

    
    sampled_df = df.sample(n=10000)
    fig2 = px.choropleth(
        sampled_df, 
        locations="State",  
        color="Severity", 
        scope="usa",
        locationmode='USA-states',
        title="Accident Severity by State"
    )
    st.plotly_chart(fig2)
    description2 = """
    ### Accident Severity by State

    This Heatmap map illustrates the severity of accidents across different states in the US based on a sample of 10,000 accident records. The severity is depicted on a scale from 1 to 4, where 1 represents the least severe accidents and 4 represents the most severe ones.
    """
    st.markdown(description2)
    
    
    st.title("Accident Locations Heatmap")
    lat_lng_pairs = list(zip(df['Start_Lat'], df['Start_Lng'].head(50)))
    heatmap = folium.Map(location=[37.0902, -95.7129], zoom_start=5)  # Centered on the US
    HeatMap(lat_lng_pairs).add_to(heatmap)
    heatmap_html = heatmap._repr_html_()
    st.components.v1.html(heatmap_html, height=600)
    


    df['Year'] = pd.to_datetime(df['Start_Time']).dt.year
    accidents_per_year = df.groupby('Year').size().reset_index(name='Accidents')
    fig = px.bar(accidents_per_year, x='Year', y='Accidents',
                title='Accident Count (2016 - 2020)',
                labels={'Accidents': 'Accidents', 'Year': 'Year'},
                text='Accidents')

    fig.update_traces(marker_color='purple', textposition='outside')
    fig.update_layout(title_text='Accident Count (2016 - 2022)', title_x=0.5,
                    xaxis_title='Year', yaxis_title='Accidents')
    st.plotly_chart(fig)

    bar_chart_description = """
    ### Accident Count (2016 - 2022)

    This bar chart visualizes the number of traffic accidents in the United States from 2016 to 2022. The data is aggregated on a yearly basis, showing the total count of accidents each year.

    """
    st.markdown(bar_chart_description)

    
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    accidents_per_month = df.groupby(['Year', 'Month']).size().reset_index(name='Accidents')
    accidents_per_month['Month'] = accidents_per_month['Month'].apply(lambda x: calendar.month_name[x])
    fig = px.bar(accidents_per_month, x='Month', y='Accidents', color='Year',
                title='Accident Count per Month (2016 - 2022)',
                labels={'Accidents': 'Accidents', 'Month': 'Month'},
                text='Accidents')

    fig.update_traces(textposition='outside')
    fig.update_layout(title_text='Accident Count per Month (2016 - 2022)', title_x=0.5,
                    xaxis_title='Month', yaxis_title='Accidents')
    st.plotly_chart(fig)
    bar_chart_description = """
    #### Accident Count per Month (2016 - 2022)

    This bar chart displays the number of traffic accidents per month across the years 2016 to 2022. Each bar represents the total count of accidents for each month, differentiated by year.
    """
    st.markdown(bar_chart_description)
        
    
    
    
    df['Week_day'] = df['Start_Time'].dt.day_of_week
    day_name_map = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    df['Week_day'] = df['Week_day'].map(day_name_map)
    accidents_per_week_day = df.groupby('Week_day').size().reset_index(name='Accidents')
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    accidents_per_week_day = accidents_per_week_day.set_index('Week_day').reindex(ordered_days).reset_index()
    fig = px.bar(accidents_per_week_day, x='Week_day', y='Accidents',
                title='Accident Count Weekly',
                labels={'Accidents': 'Accidents', 'Week_day': 'Day of the Week'},
                text='Accidents')
    fig.update_traces(marker_color='yellow', textposition='outside')
    fig.update_layout(title_text='Accident Count Per Day', title_x=0.5,
                    xaxis_title='Day of the Week', yaxis_title='Accidents')
    st.title("Weekly Accident Count Analysis")
    st.write("This bar chart shows the number of traffic accidents for each day of the week.")
    st.plotly_chart(fig)

    
    
    
    
    df['Hour'] = df['Start_Time'].dt.hour
    accidents_per_hour = df.groupby('Hour').size().reset_index(name='Accidents')
    fig = px.bar(accidents_per_hour, x='Hour', y='Accidents',
                title='Accidents by Hour of the Day',
                labels={'Accidents': 'Number of Accidents', 'Hour': 'Hour of the Day'},
                text='Accidents')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_text='Accidents by Hour of the Day', title_x=0.5,
                    xaxis_title='Hour of the Day', yaxis_title='Number of Accidents',
                    xaxis=dict(tickmode='linear', tickvals=list(range(24))))
    st.title("Accidents by Hour of the Day")
    st.write("This bar chart shows the number of traffic accidents that occurred each hour of the day.")
    st.plotly_chart(fig)


    
    
    monthly_accidents = df.groupby(['City', 'Year', 'Month']).size().reset_index(name='Accidents')
    total_accidents_per_city = monthly_accidents.groupby('City')['Accidents'].sum().reset_index()
    top_20_cities = total_accidents_per_city.nlargest(10, 'Accidents')['City']
    filtered_data = monthly_accidents[monthly_accidents['City'].isin(top_20_cities)]
    pivot_table = filtered_data.pivot_table(index='City', columns='Month', values='Accidents', aggfunc='sum', fill_value=0)
    fig = px.imshow(pivot_table,
                    color_continuous_scale='YlGnBu',
                    labels={'color': 'Accidents'},
                    title='Monthly Accidents for Top 10 Cities')
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='City',
        xaxis=dict(tickmode='array', tickvals=list(range(12)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    )
    st.title("Monthly Accidents for Top 10 Cities")
    st.write("This heatmap shows the distribution of traffic accidents for the top 10 cities, broken down by month.")
    st.plotly_chart(fig)


    
    
    low_severity_condition = df['Severity'] == 1  # Assuming '1' represents low severity
    low_visibility_condition = df['Visibility(mi)'] < 1  # Low visibility defined as less than 1 mile
    filtered_df = df[low_severity_condition & low_visibility_condition]
    accidents_per_city = filtered_df.groupby('City').size().reset_index(name='Accidents')
    top_20_cities = accidents_per_city.nlargest(20, 'Accidents')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Accidents', y='City', data=top_20_cities, palette='coolwarm', ax=ax)
    ax.set_xlabel('Number of Accidents')
    ax.set_ylabel('City')
    ax.set_title('Low Severity Accidents Due to Low Visibility in Top 20 Cities')
    st.title("Low Severity Accidents Due to Low Visibility")
    st.write("This bar plot shows the number of low severity accidents caused by low visibility (less than 1 mile) across the top 20 cities.")
    st.pyplot(fig)

    
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Distance(mi)'], bins=30, color='green', kde=False, binrange=(0, 7.0), ax=ax)

    ax.set_xlabel("Distance in miles")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Road Length Affected by Accident")
    average_distance = df["Distance(mi)"].mean()
    st.title("Histogram of Road Length Affected by Accidents")
    st.write("This histogram shows the distribution of road lengths affected by accidents. The bins represent the range of distances, and the height of each bar indicates the count of accidents within that range.")
    st.pyplot(fig)
    st.write(f"Average road length affected is {average_distance:.2f} miles")

    
    
    
    df['Temperature(C)'] = (df['Temperature(F)'] - 32) * 5 / 9
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["Temperature(C)"].dropna(), 
                color="dodgerblue", 
                bins=30, 
                kde=True, 
                alpha=0.6, 
                linewidth=2, 
                ax=ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Temperatures (Celsius)")
    st.title("Temperature Distribution in Celsius")
    st.write("This plot shows the distribution of temperatures in Celsius across the dataset. The histogram represents the frequency of temperatures within each range, while the KDE (Kernel Density Estimate) curve illustrates the density distribution.")
    st.pyplot(fig)

    
    
    
    st.title("Accident Analysis by Severity")
    st.write("""The graph displays the top 25 weather conditions associated with traffic accidents for the selected severity level, highlighting the frequency of each condition.
            """)
    severity = st.radio(
        "Select Accident Severity",
        options=[1, 2, 3, 4]
    )
    severity_df = df[df['Severity'] == severity]
    weather_counts = severity_df['Weather_Condition'].value_counts().head(25)
    weather_counts_df = weather_counts.reset_index()
    weather_counts_df.columns = ['Weather_Condition', 'Accident_Count']
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Accident_Count', y='Weather_Condition', data=weather_counts_df, hue='Weather_Condition', dodge=False, palette='viridis')
    plt.xlabel('Accident Count')
    plt.ylabel('Weather Condition')
    plt.title(f'Top 25 Weather Conditions for Accidents of Severity {severity}')
    plt.legend([],[], frameon=False)
    st.pyplot(plt)

    
    
    
    def plot_severity_distribution(condition):
        severity_counts = df.loc[df["Weather_Condition"] == condition]["Severity"].value_counts()
        description = """
        ### Overview
        The graphs provide insights into the distribution of accident severity under different weather conditions. For each selected weather condition, the visualization consists of two components:

        - **Bar Plot**: Displays the count of accidents categorized by severity levels. This plot helps in understanding how frequently different severity levels occur under specific weather conditions.

        - **Pie Chart**: Shows the proportion of each severity level as a percentage of the total accidents for the selected weather condition. This chart highlights the relative impact of each severity level within the context of the chosen weather.
        """
        st.markdown(description)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Accident Severity Under {condition}", fontsize=16)
        axes[0].bar(severity_counts.index, severity_counts.values, color='g' if condition == "Fog" else 'r', width=0.5)
        axes[0].set_xlabel("Severity", fontsize=16)
        axes[0].set_ylabel("Accidents Count", fontsize=16)
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[1].pie(severity_counts, labels=severity_counts.index, autopct="%1.1f%%", colors=sns.color_palette("viridis", len(severity_counts)))

        return fig
    st.title("Accident Severity by Weather Condition")
    selected_condition = st.selectbox(
        "Choose a weather condition:",
        ["Light Rain", "Rain", "Heavy Rain","Fog", "Snow"]
    )
    fig = plot_severity_distribution(selected_condition)
    st.pyplot(fig)



    
    
    st.title("Accident Locations Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df['Start_Lng'], y=df['Start_Lat'], size=0.001, alpha=0.5, color='purple', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Accident Locations')
    st.pyplot(fig)
    st.markdown("""
    ### Accident Locations Scatter Plot

    This scatter plot visualizes the geographical distribution of accident locations across the United States. 

    - **X-Axis (Longitude)**: Represents the longitude of accident locations.
    - **Y-Axis (Latitude)**: Represents the latitude of accident locations.
    - **Plot Details**: Each point on the plot corresponds to an accident, with points scattered based on their geographical coordinates. The plot uses a purple color for the markers and includes a level of transparency (alpha) to help visualize dense areas more clearly.
    """)




    factors = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
           'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
           'Turning_Loop']
    st.title("Accident Severity Analysis by Road Factors")
    selected_factor = st.selectbox('Select a road factor:', factors)
    if (df[selected_factor] == True).sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Accident Severity Near " + selected_factor, fontsize=16)
        severity_counts = df.loc[df[selected_factor] == True]['Severity'].value_counts()
        axes[0].bar(severity_counts.index, severity_counts.values, color='y', width=0.5)
        axes[0].set_xlabel("Severity", fontsize=16)
        axes[0].set_ylabel("Accident Count", fontsize=16)
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[1].pie(severity_counts, labels=severity_counts.index, autopct="%1.0f%%", colors=sns.color_palette("viridis", len(severity_counts)))
        st.pyplot(fig)
    else:
        st.write(f"No accidents recorded for the selected factor: {selected_factor}.")


    
    
    
    weather_conditions = ["Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)"]
    df_filtered = df[['Severity'] + weather_conditions].dropna()
    df_filtered[weather_conditions] = df_filtered[weather_conditions].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df_filtered.corr()
    st.title('Correlation Heatmap Between Weather Conditions and Accident Severity')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    plt.title('Correlation Heatmap Between Weather Conditions and Accident Severity', fontsize=16)
    st.pyplot(fig)

    
    
    st.title('Accident Distribution by Sunrise/Sunset')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Sunrise_Sunset', data=df, ax=ax, hue='Sunrise_Sunset')
    ax.set_title('Accident Distribution by Sunrise/Sunset')
    ax.set_xlabel('Sunrise/Sunset')
    ax.set_ylabel('Number of Accidents')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    
    
    
    top_cities = df['City'].value_counts().nlargest(6).index
    df_top_cities = df[df['City'].isin(top_cities)]
    colors = {1: 'blue', 2: 'red', 3: 'yellow', 4: 'purple'}
    st.title('Accident Scatter Plots by City')
    selected_city = st.radio("Select a City", options=top_cities)
    city_data = df_top_cities[df_top_cities['City'] == selected_city]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='Severity', data=city_data, palette=colors, ax=ax)
    ax.set_title(f'Accidents in {selected_city}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(title='Severity', title_fontsize='13', fontsize='12')
    st.pyplot(fig)

    
    
    
    df = df[['Severity', 'Temperature(F)', 'Visibility(mi)']].dropna()
    st.title('3D Scatter Plot of Accident Severity vs. Temperature and Visibility')
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = {1: 'blue', 2: 'red', 3: 'yellow', 4: 'purple'}
    sc = ax.scatter(df['Temperature(F)'], df['Visibility(mi)'], df['Severity'], 
                    c=df['Severity'], cmap='viridis', edgecolor='w', s=60)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Severity')
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'])
    ax.set_xlabel('Temperature (°F)')
    ax.set_ylabel('Visibility (mi)')
    ax.set_zlabel('Severity')
    ax.set_title('3D Scatter Plot of Accident Severity vs. Temperature and Visibility')
    st.pyplot(fig)

    
        
        
        
elif st.session_state.page == "Hypothesis":
    def test_avg_severity(df):
        severity_mean = df['Severity'].mean()
        t_stat, p_value = stats.ttest_1samp(df['Severity'], 2.5)
        return severity_mean, t_stat, p_value

    def test_impact_on_traffic(df):
        df = df[df['Distance(mi)'].notna()]
        mean_distance = df['Distance(mi)'].mean()
        t_stat, p_value = stats.ttest_1samp(df['Distance(mi)'], 1)
        return mean_distance, t_stat, p_value

    def test_weather_impact(df):
        df = df[['Weather_Condition', 'Severity']].dropna()
        weather_conditions = df['Weather_Condition'].unique()
        severity_means = df.groupby('Weather_Condition')['Severity'].mean()
        f_stat, p_value = stats.f_oneway(*(df[df['Weather_Condition'] == condition]['Severity'] for condition in weather_conditions))
        return severity_means, f_stat, p_value

    st.title('Accident Data Analysis')
    selected_button = st.sidebar.radio(
        "Select an Analysis",
        ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"]
    )

    if selected_button == "Hypothesis 1":
        st.subheader('Hypothesis 1: Average Severity of Car Accidents')
        severity_mean, t_stat, p_value = test_avg_severity(df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Severity'], bins=range(1, 5), color='blue', kde=False, ax=ax)
        ax.axvline(severity_mean, color='red', linestyle='--', label=f'Mean Severity: {severity_mean:.2f}')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Number of Accidents')
        ax.set_title('Distribution of Accident Severity')
        ax.legend()
        st.pyplot(fig)
        
        st.write("**Explanation:**")
        st.write("**Diagram:** A histogram of accident severity with a red dashed line showing the mean severity.")
        st.write("**Purpose:** To visualize the distribution of accident severities and compare it with the mean threshold of 2.5.")
        
        st.write(f'Average Severity: {severity_mean:.2f}')
        st.write(f'T-Statistic: {t_stat:.2f}')
        st.write(f'P-Value: {p_value:.2f}')
        if p_value < 0.05:
            st.write("**Reject the Null Hypothesis:** The average severity of car accidents is significantly greater than 2.5.")
        else:
            st.write("**Fail to Reject the Null Hypothesis:** There is not enough evidence to conclude that the average severity is greater than 2.5.")

    elif selected_button == "Hypothesis 2":
        st.subheader('Hypothesis 2: Impact on Traffic')
        mean_distance, t_stat, p_value = test_impact_on_traffic(df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Distance(mi)'], bins=30, color='green', kde=False, ax=ax)
        ax.axvline(mean_distance, color='red', linestyle='--', label=f'Mean Distance: {mean_distance:.2f} miles')
        ax.set_xlabel('Distance (miles)')
        ax.set_ylabel('Number of Accidents')
        ax.set_title('Distribution of Distance Affected by Accidents')
        ax.legend()
        st.pyplot(fig)
        
        st.write("**Explanation:**")
        st.write("**Diagram:** A histogram of the distance affected by accidents with a red dashed line showing the mean distance.")
        st.write("**Purpose:** To visualize the distribution of the distance affected by accidents and check if the mean is below the threshold of 1 mile.")
        
        st.write(f'Average Distance Affected: {mean_distance:.2f} miles')
        st.write(f'T-Statistic: {t_stat:.2f}')
        st.write(f'P-Value: {p_value:.2f}')
        if p_value < 0.05:
            st.write("**Reject the Null Hypothesis:** The average distance affected is significantly less than one mile.")
        else:
            st.write("**Fail to Reject the Null Hypothesis:** There is not enough evidence to conclude that the average distance affected is less than one mile.")

    elif selected_button == "Hypothesis 3":
        st.subheader('Hypothesis 3: Impact of Weather Conditions on Traffic')
        severity_means, f_stat, p_value = test_weather_impact(df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=severity_means.index, y=severity_means.values, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Average Severity')
        ax.set_title('Average Severity by Weather Condition')
        st.pyplot(fig)
        
        st.write("**Explanation:**")
        st.write("**Diagram:** A barplot showing the average severity of accidents for different weather conditions.")
        st.write("**Purpose:** To visualize the variation in severity across different weather conditions and determine if weather has a significant impact on severity.")
        
        st.write(f'F-Statistic: {f_stat:.2f}')
        st.write(f'P-Value: {p_value:.2f}')
        st.write("Mean Severity by Weather Condition:")
        st.write(severity_means)
        if p_value < 0.05:
            st.write("**Reject the Null Hypothesis:** Different weather conditions have a significant impact on the severity of accidents.")
        else:
            st.write("**Fail to Reject the Null Hypothesis:** There is not enough evidence to conclude that weather conditions affect the severity of accidents.")
