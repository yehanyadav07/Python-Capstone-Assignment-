# main.py

import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 0. Setup and Configuration ---
# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input/output directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the output directory exists

# --- Task 1: Data Ingestion and Validation ---

def ingest_and_validate_data():
    """Reads multiple CSV files, handles errors, and combines them into one clean DataFrame."""
    logging.info("Starting Task 1: Data Ingestion and Validation.")
    df_combined = pd.DataFrame()
    
    if not DATA_DIR.is_dir():
        logging.error(f"Data directory not found at {DATA_DIR}. Please create it and add CSV files.")
        return None
        
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in {DATA_DIR}.")
        return None
        
    for file_path in csv_files:
        file_name = file_path.name
        try:
            # 1. Infer metadata from filename (Assumes format like 'building_A_month.csv')
            parts = file_name.replace(".csv", "").split("_")
            building_name = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
            
            # 2. Read file, skipping bad lines
            # Assuming the raw files have at least two columns: [Timestamp, kwh]
            df = pd.read_csv(
                file_path, 
                on_bad_lines='skip', # Handles corrupt data by skipping bad lines
                low_memory=False
            )
            
            # 3. Rename/Assign core columns based on expected structure
            if len(df.columns) < 2:
                 raise ValueError("File must contain at least two columns: Timestamp and kwh.")
                 
            df.rename(columns={df.columns[0]: 'Timestamp', df.columns[1]: 'kwh'}, inplace=True)
            df['Building'] = building_name
            
            df_combined = pd.concat([df_combined, df], ignore_index=True)
            logging.info(f"Successfully ingested and validated: {file_name}")

        except FileNotFoundError:
            logging.error(f"Missing file error: {file_name}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {file_name}: {e}")

    # 4. Clean and Prepare the combined data
    if not df_combined.empty:
        df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], errors='coerce')
        df_combined['kwh'] = pd.to_numeric(df_combined['kwh'], errors='coerce')
        df_combined.dropna(subset=['Timestamp', 'kwh'], inplace=True)
        df_combined.set_index('Timestamp', inplace=True)
        logging.info("Task 1 complete: Data combined and cleaned.")
        
    return df_combined

# --- Task 2: Core Aggregation Logic ---

def calculate_daily_totals(df):
    """Calculates total daily consumption per building."""
    logging.info("Calculating daily totals.")
    # Group by Building, then resample the Timestamp index to daily ('D') frequency and sum the kwh
    df_daily = df.groupby('Building')['kwh'].resample('D').sum().reset_index()
    df_daily.rename(columns={'kwh': 'Daily_kwh_Total'}, inplace=True)
    return df_daily

def calculate_weekly_aggregates(df):
    """Calculates total weekly consumption per building."""
    logging.info("Calculating weekly totals.")
    # Group by Building, then resample the Timestamp index to weekly ('W') frequency and sum the kwh
    df_weekly = df.groupby('Building')['kwh'].resample('W').sum().reset_index()
    df_weekly.rename(columns={'kwh': 'Weekly_kwh_Total'}, inplace=True)
    return df_weekly

def building_wise_summary(df):
    """Calculates mean, min, max, and total consumption for each building."""
    logging.info("Calculating building-wise summary.")
    building_summary_df = df.groupby('Building')['kwh'].agg(
        Mean_kwh='mean',
        Min_kwh='min',
        Max_kwh='max',
        Total_kwh='sum'
    ).reset_index()
    
    # Store results in a dictionary (as per requirement)
    summary_dict = building_summary_df.set_index('Building').T.to_dict('dict')
    
    return building_summary_df, summary_dict

# --- Task 3: Object-Oriented Modeling ---

class MeterReading:
    """Represents a single meter reading at a specific time."""
    def __init__(self, timestamp, kwh):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

class Building:
    """Represents a building with a collection of meter readings."""
    def __init__(self, name):
        self.name = name
        self.meter_readings = []
        
    def add_reading(self, timestamp, kwh):
        """Adds a new MeterReading instance to the building."""
        try:
            reading = MeterReading(timestamp, kwh)
            self.meter_readings.append(reading)
        except Exception:
            # Skip invalid reading, logging handles detailed errors in Task 1
            pass 
        
    def calculate_total_consumption(self):
        """Calculates the sum of all kwh readings."""
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self):
        """Generates a simple usage report string."""
        total = self.calculate_total_consumption()
        
        report = (
            f"--- Report for Building {self.name} ---\n"
            f"Total Consumption: {total:,.2f} kWh\n"
        )
        return report


class BuildingManager:
    """Manages all Building objects and provides campus-wide statistics."""
    def __init__(self):
        self.buildings = {} # Stores Building objects: {'name': Building_Object}

    def add_data_from_dataframe(self, df_combined):
        """Populates the manager with data from the Task 1 DataFrame."""
        logging.info("Task 3: Populating OOP model.")
        
        # Reset index to iterate over Timestamp and Building columns easily
        df_iter = df_combined.reset_index()
        
        for name in df_iter['Building'].unique():
            self.buildings[name] = Building(name)
        
        # Iterate over the rows and add readings to the correct building object
        for _, row in df_iter.iterrows():
            building_name = row['Building']
            # Using the original Timestamp and kwh values
            self.buildings[building_name].add_reading(row['Timestamp'], row['kwh'])

    def calculate_campus_total(self):
        """Calculates total consumption across all buildings."""
        return sum(b.calculate_total_consumption() for b in self.buildings.values())


# --- Task 4: Visual Output with Matplotlib ---

def generate_dashboard_plots(df_daily, df_weekly, df_combined):
    """Generates a three-panel dashboard and saves it as a PNG."""
    logging.info("Starting Task 4: Generating visual output.")
    
    # --- Data Prep for Scatter Plot ---
    # Resample to hourly to find the hour of the day with high usage
    df_hourly = df_combined.groupby('Building')['kwh'].resample('H').sum().reset_index()
    df_hourly['Hour'] = df_hourly['Timestamp'].dt.hour
    
    # Calculate the average hourly consumption per building for stability in the scatter plot
    df_peak = df_hourly.groupby(['Building', 'Hour'])['kwh'].mean().reset_index()
    
    # Prepare data for Bar Chart (Average Weekly Usage)
    avg_weekly = df_weekly.groupby('Building')['Weekly_kwh_Total'].mean().reset_index()

    # --- Create Figure and Subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.style.use('ggplot')
    
    # Plot 1: Trend Line (Daily Consumption)
    sns.lineplot(data=df_daily, x='Timestamp', y='Daily_kwh_Total', hue='Building', ax=axes[0])
    axes[0].set_title('1. Daily Total Energy Consumption Trend')
    axes[0].set_ylabel('Total kWh')
    axes[0].set_xlabel('Date')
    
    # Plot 2: Bar Chart (Average Weekly Usage)
    sns.barplot(data=avg_weekly, x='Building', y='Weekly_kwh_Total', ax=axes[1], palette='viridis')
    axes[1].set_title('2. Comparison of Average Weekly Consumption')
    axes[1].set_ylabel('Average Weekly kWh')
    axes[1].set_xlabel('Building Name')

    # Plot 3: Scatter Plot (Average Consumption by Hour of Day)
    sns.scatterplot(data=df_peak, x='Hour', y='kwh', hue='Building', ax=axes[2], s=100)
    axes[2].set_title('3. Average Consumption by Hour of Day (Peak Load Indicator)')
    axes[2].set_ylabel('Average Hourly kWh')
    axes[2].set_xlabel('Hour of Day (0-23)')
    axes[2].set_xticks(range(0, 24, 2))
    
    # Final layout adjustments
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dashboard.png")
    logging.info("Task 4 complete: Dashboard image saved to output/dashboard.png")
    plt.close(fig)

# --- Task 5: Persistence and Executive Summary ---

def persist_data(df_combined, df_summary):
    """Exports processed data and summary statistics to CSV files."""
    logging.info("Starting Task 5: Data Persistence.")
    
    # 1. Export Final processed dataset
    df_combined.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv")
    logging.info("Exported cleaned_energy_data.csv")
    
    # 2. Export Summary stats
    df_summary.to_csv(OUTPUT_DIR / "building_summary.csv", index=False)
    logging.info("Exported building_summary.csv")

def generate_executive_summary(manager, df_daily, df_summary):
    """Creates a concise written report (summary.txt) based on analysis."""
    
    # 1. Calculate Core Metrics
    total_campus_consumption = manager.calculate_campus_total()
    
    # Highest consuming building
    highest_consumer_row = df_summary.loc[df_summary['Total_kwh'].idxmax()]
    highest_consumer = highest_consumer_row['Building']
    highest_kwh = highest_consumer_row['Total_kwh']
    
    # Peak Load Time (Find the hour of the day with the highest *average* usage)
    df_hourly_overall = df_daily.reset_index().set_index('Timestamp')['Daily_kwh_Total'].resample('H').sum()
    peak_hour_overall = df_hourly_overall.groupby(df_hourly_overall.index.hour).mean().idxmax()
    
    # Weekly/Daily Trends Assessment
    daily_mean = df_daily['Daily_kwh_Total'].mean()
    daily_std = df_daily['Daily_kwh_Total'].std()
    
    if daily_std / daily_mean > 0.2:
        trend_note = "Significant variability in daily consumption, suggesting large operational swings or scheduling inefficiencies."
    else:
        trend_note = "Relatively stable daily consumption, indicating consistent usage patterns."
    
    # 2. Compile Report
    summary_text = (
        "=========================================\n"
        "   ENERGY USAGE EXECUTIVE SUMMARY REPORT\n"
        "=========================================\n\n"
        
        f"**1. Total Campus Consumption:**\n"
        f"   Total recorded consumption: {total_campus_consumption:,.2f} kWh\n\n"
        
        f"**2. Highest Consuming Building:**\n"
        f"   Building: **{highest_consumer}**\n"
        f"   Total Consumption: {highest_kwh:,.2f} kWh (See building_summary.csv for details)\n\n"
        
        f"**3. Peak Load Time:**\n"
        f"   The peak average load time across the campus is **Hour {peak_hour_overall}:00**.\n"
        f"   *Recommendation: Investigate energy-intensive activities and equipment scheduling during this hour.*\n\n"
        
        f"**4. Usage Trends:**\n"
        f"   Analysis shows {trend_note}\n"
        f"   Refer to 'dashboard.png' for visualization of daily and weekly trends."
    )
    
    # 3. Save Report
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        f.write(summary_text)
        
    print("\n" + summary_text)
    logging.info("Task 5 complete: Executive summary saved to output/summary.txt")


# --- Main Execution Block ---

def main():
    """Runs the complete energy dashboard pipeline."""
    logging.info("--- Starting Energy Dashboard Pipeline ---")
    
    # 1. Ingestion and Validation
    df_combined = ingest_and_validate_data()
    if df_combined is None or df_combined.empty:
        logging.error("Pipeline aborted: Cannot proceed without valid data.")
        return

    # 2. Core Aggregation
    df_daily = calculate_daily_totals(df_combined)
    df_weekly = calculate_weekly_aggregates(df_combined)
    df_summary, summary_dict = building_wise_summary(df_combined)
    
    # 3. Object-Oriented Modeling
    manager = BuildingManager()
    # Note: df_combined is reset when it was cleaned in Task 1, so we must reset index for merging/grouping by 'Building'
    manager.add_data_from_dataframe(df_combined)

    # 4. Visual Output
    generate_dashboard_plots(df_daily, df_weekly, df_combined)
    
    # 5. Persistence and Executive Summary
    persist_data(df_combined.reset_index(), df_summary) # Reset index for clean CSV export
    generate_executive_summary(manager, df_daily, df_summary)
    
    logging.info("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
