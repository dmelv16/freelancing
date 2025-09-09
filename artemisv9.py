#!/usr/bin/env python3
"""
Enhanced Bus Monitor Analysis Dashboard - Multi-Level Analysis Version
Provides station-level, save-level, and file-level analysis with drill-down capabilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EnhancedBusMonitorDashboard:
    def __init__(self):
        """
        Initialize the Enhanced Bus Monitor Dashboard
        """
        # CONFIGURATION - Change these paths as needed
        self.csv_folder = Path("./csv_data")  # <-- UPDATE THIS PATH
        self.lookup_csv_path = Path("./message_lookup.csv")  # <-- PATH TO LOOKUP CSV
        self.output_folder = Path("./bus_monitor_output")  # <-- OUTPUT FOLDER
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Analysis results storage
        self.bus_flip_issues = []
        self.data_changes = []
        self.flip_statistics = defaultdict(lambda: defaultdict(int))
        self.all_files_data = []
        self.header_issues = []
        self.file_message_summary = []
        self.detailed_flip_data = []
        
        # New hierarchical storage
        self.station_summary = []
        self.save_summary = []
        self.station_save_matrix = []
        self.message_type_by_station = []
        self.flip_patterns = []
        self.time_series_data = []
        
        # DataFrames for analysis
        self.df_summary = None
        self.df_flips = None
        self.df_changes = None
        self.df_headers = None
        self.df_file_messages = None
        self.df_detailed_flips = None
        
        # New DataFrames for hierarchical analysis
        self.df_station_summary = None
        self.df_save_summary = None
        self.df_station_save_matrix = None
        self.df_message_type_station = None
        self.df_flip_patterns = None
        self.df_time_series = None
        
        # Load message type to header lookup
        self.message_header_lookup = self.load_message_lookup()
    
    def load_message_lookup(self):
        """
        Load the message type to header lookup table
        """
        lookup = defaultdict(list)
        
        if self.lookup_csv_path.exists():
            try:
                df_lookup = pd.read_csv(self.lookup_csv_path)
                print(f"Loaded message lookup from {self.lookup_csv_path}")
                
                for msg_type, group in df_lookup.groupby('message_type'):
                    lookup[msg_type] = group['header'].tolist()
                
                print(f"  Loaded {len(lookup)} message types with header mappings")
                
            except Exception as e:
                print(f"Warning: Could not load message lookup: {e}")
        else:
            print(f"Note: No message lookup file found at {self.lookup_csv_path}")
            print("  Header validation will be skipped")
        
        return lookup
    
    def validate_header(self, msg_type: str, actual_header: str):
        """
        Validate if the actual header matches expected headers for message type
        """
        if not self.message_header_lookup or msg_type is None:
            return True, []
        
        expected_headers = self.message_header_lookup.get(msg_type, [])
        
        if not expected_headers:
            return True, []
        
        actual_clean = str(actual_header).strip().upper() if pd.notna(actual_header) else ""
        expected_clean = [str(h).strip().upper() for h in expected_headers]
        
        is_valid = actual_clean in expected_clean
        
        return is_valid, expected_headers
    
    def parse_filename(self, filename: str):
        """
        Parse filename to extract unit_id, station, save, and station_num
        """
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            return {
                'unit_id': parts[0],
                'station': parts[1],
                'save': parts[2],
                'station_num': parts[3] if len(parts) > 3 else 'unknown',
                'filename': filename
            }
        return None
    
    def extract_message_type(self, decoded_desc: str):
        """
        Extract message type from decoded description
        """
        if pd.isna(decoded_desc):
            return None, 0
            
        pattern = r'\((\d+)-\[([^\]]+)\]-(\d+)\)'
        match = re.search(pattern, str(decoded_desc))
        
        if match:
            msg_type = match.group(2)
            end_num = int(match.group(3))
            return msg_type, end_num
        
        return None, 0
    
    def detect_bus_flips(self, df: pd.DataFrame, file_info: dict):
        """
        Detect rapid bus flips with matching decoded_description
        """
        flips = []
        
        if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
            return flips
            
        df = df.copy()
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            return flips
        
        # Vectorized operations
        df['prev_bus'] = df['bus'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        df['prev_decoded'] = df['decoded_description'].shift(1)
        df['time_diff_ms'] = (df['timestamp'] - df['prev_timestamp']) * 1000
        
        # Find flips with matching decoded_description
        mask = (
            (df['bus'] != df['prev_bus']) & 
            (df['time_diff_ms'] < 100) & 
            (df['time_diff_ms'].notna()) &
            (df['decoded_description'] == df['prev_decoded'])
        )
        
        flip_indices = df[mask].index.tolist()
        
        for idx in flip_indices:
            curr_row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            msg_type, _ = self.extract_message_type(curr_row.get('decoded_description'))
            
            # Check header validation
            header_issue = False
            if 'data01' in df.columns and msg_type:
                is_valid_prev, expected = self.validate_header(msg_type, prev_row.get('data01'))
                is_valid_curr, _ = self.validate_header(msg_type, curr_row.get('data01'))
                
                if not is_valid_prev or not is_valid_curr:
                    header_issue = True
                    self.header_issues.append({
                        'filename': file_info['filename'],
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'timestamp': prev_row['timestamp'],
                        'msg_type': msg_type,
                        'actual_header_busA': prev_row.get('data01') if prev_row['bus'] == 'A' else curr_row.get('data01'),
                        'actual_header_busB': prev_row.get('data01') if prev_row['bus'] == 'B' else curr_row.get('data01'),
                        'expected_headers': ', '.join(map(str, expected))
                    })
            
            # Create flip record with enhanced metadata
            flip_info = {
                'filename': file_info['filename'],
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'station_save': f"{file_info['station']}_{file_info['save']}",  # New field
                'timestamp': prev_row['timestamp'],
                'time_diff_ms': round(curr_row['time_diff_ms'], 2),
                'bus_transition': f"{prev_row['bus']} to {curr_row['bus']}",
                'msg_type': msg_type,
                'decoded_description': curr_row['decoded_description'],
                'data01_prev': prev_row.get('data01', ''),
                'data01_curr': curr_row.get('data01', ''),
                'header_issue': header_issue
            }
            
            flips.append(flip_info)
            
            # Track data changes
            data_changes = self.compare_data_words(prev_row, curr_row)
            for col, change in data_changes.items():
                self.data_changes.append({
                    'filename': file_info['filename'],
                    'unit_id': file_info['unit_id'],
                    'station': file_info['station'],
                    'save': file_info['save'],
                    'station_save': f"{file_info['station']}_{file_info['save']}",
                    'timestamp': prev_row['timestamp'],
                    'msg_type': msg_type,
                    'decoded_description': curr_row['decoded_description'],
                    'data_column': col,
                    'value_busA': change['before'] if prev_row['bus'] == 'A' else change['after'],
                    'value_busB': change['before'] if prev_row['bus'] == 'B' else change['after']
                })
            
            # Store complete row data
            detailed_flip = {
                'filename': file_info['filename'],
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'flip_timestamp': prev_row['timestamp'],
                'bus_transition': f"{prev_row['bus']} to {curr_row['bus']}",
                'decoded_description': curr_row['decoded_description'],
                'msg_type': msg_type
            }
            
            # Add all data columns
            for col in df.columns:
                if col.startswith('data') or col in ['timestamp', 'bus', 'decoded_description']:
                    detailed_flip[f'{col}_busA'] = prev_row[col] if prev_row['bus'] == 'A' else curr_row[col]
                    detailed_flip[f'{col}_busB'] = prev_row[col] if prev_row['bus'] == 'B' else curr_row[col]
            
            self.detailed_flip_data.append(detailed_flip)
        
        return flips
    
    def compare_data_words(self, row1: pd.Series, row2: pd.Series):
        """
        Compare data word columns between two rows
        """
        changes = {}
        
        data_cols = [col for col in row1.index if col.startswith('data') and 
                    len(col) > 4 and col[4:].replace('0', '').isdigit()]
        
        for col in data_cols:
            if col in row2.index:
                val1, val2 = row1[col], row2[col]
                
                if pd.isna(val1) and pd.isna(val2):
                    continue
                    
                if str(val1) != str(val2):
                    changes[col] = {
                        'before': val1,
                        'after': val2,
                        'bus_before': row1['bus'],
                        'bus_after': row2['bus']
                    }
        
        return changes
    
    def analyze_file_messages(self, df: pd.DataFrame, file_info: dict):
        """
        Analyze all messages in a file
        """
        if 'decoded_description' not in df.columns:
            return
        
        unique_messages = df['decoded_description'].dropna().unique()
        
        for decoded_desc in unique_messages:
            msg_type, _ = self.extract_message_type(decoded_desc)
            msg_data = df[df['decoded_description'] == decoded_desc]
            
            self.file_message_summary.append({
                'filename': file_info['filename'],
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'decoded_description': decoded_desc,
                'msg_type': msg_type or 'unknown',
                'total_count': len(msg_data),
                'bus_a_count': len(msg_data[msg_data['bus'] == 'A']),
                'bus_b_count': len(msg_data[msg_data['bus'] == 'B']),
                'has_flips': decoded_desc in [f['decoded_description'] for f in self.bus_flip_issues]
            })
    
    def process_csv(self, csv_path: Path):
        """
        Process a single CSV file
        """
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            return None
        
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None
            
            flips = self.detect_bus_flips(df, file_info)
            if flips:
                self.bus_flip_issues.extend(flips)
            
            self.analyze_file_messages(df, file_info)
            
            bus_counts = df['bus'].value_counts()
            
            # Calculate time span if timestamp exists
            time_span_hours = 0
            if 'timestamp' in df.columns and len(df) > 0:
                df_clean = df.dropna(subset=['timestamp'])
                if len(df_clean) > 0:
                    time_span_hours = (df_clean['timestamp'].max() - df_clean['timestamp'].min()) / 3600
            
            return {
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'station_save': f"{file_info['station']}_{file_info['save']}",
                'total_rows': len(df),
                'bus_flips': len(flips),
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0),
                'unique_messages': df['decoded_description'].nunique() if 'decoded_description' in df.columns else 0,
                'time_span_hours': round(time_span_hours, 2),
                'flips_per_hour': round(len(flips) / time_span_hours, 2) if time_span_hours > 0 else 0
            }
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            return None
    
    def aggregate_station_data(self):
        """
        Aggregate data at the station level
        """
        if not self.df_summary.empty:
            # Station-level aggregation
            station_agg = self.df_summary.groupby('station').agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'save': 'nunique',
                'unit_id': 'nunique',
                'time_span_hours': 'sum'
            }).reset_index()
            
            station_agg.columns = ['station', 'total_rows', 'total_flips', 'bus_a_total', 
                                  'bus_b_total', 'total_unique_messages', 'file_count', 
                                  'save_count', 'unit_count', 'total_hours']
            
            station_agg['flip_rate_per_1000'] = (station_agg['total_flips'] / station_agg['total_rows'] * 1000).round(2)
            station_agg['flips_per_hour'] = (station_agg['total_flips'] / station_agg['total_hours']).round(2)
            station_agg['bus_balance_ratio'] = (station_agg['bus_a_total'] / (station_agg['bus_a_total'] + station_agg['bus_b_total'])).round(3)
            
            # Calculate severity score (higher is worse)
            station_agg['severity_score'] = (
                station_agg['flip_rate_per_1000'] * 0.4 +
                abs(station_agg['bus_balance_ratio'] - 0.5) * 100 * 0.3 +
                station_agg['flips_per_hour'] * 0.3
            ).round(2)
            
            self.station_summary = station_agg.to_dict('records')
            self.df_station_summary = station_agg.sort_values('severity_score', ascending=False)
    
    def aggregate_save_data(self):
        """
        Aggregate data at the save level within each station
        """
        if not self.df_summary.empty:
            # Save-level aggregation
            save_agg = self.df_summary.groupby(['station', 'save']).agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'unit_id': 'nunique',
                'time_span_hours': 'sum'
            }).reset_index()
            
            save_agg.columns = ['station', 'save', 'total_rows', 'total_flips', 
                               'bus_a_total', 'bus_b_total', 'total_unique_messages', 
                               'file_count', 'unit_count', 'total_hours']
            
            save_agg['flip_rate_per_1000'] = (save_agg['total_flips'] / save_agg['total_rows'] * 1000).round(2)
            save_agg['flips_per_hour'] = (save_agg['total_flips'] / save_agg['total_hours']).round(2)
            save_agg['bus_balance_ratio'] = (save_agg['bus_a_total'] / (save_agg['bus_a_total'] + save_agg['bus_b_total'])).round(3)
            
            # Calculate severity score
            save_agg['severity_score'] = (
                save_agg['flip_rate_per_1000'] * 0.4 +
                abs(save_agg['bus_balance_ratio'] - 0.5) * 100 * 0.3 +
                save_agg['flips_per_hour'] * 0.3
            ).round(2)
            
            self.save_summary = save_agg.to_dict('records')
            self.df_save_summary = save_agg.sort_values('severity_score', ascending=False)
    
    def create_station_save_matrix(self):
        """
        Create a matrix view of stations vs saves with flip counts
        """
        if self.df_flips is not None and not self.df_flips.empty:
            # Create pivot table
            matrix = self.df_flips.groupby(['station', 'save']).size().reset_index(name='flip_count')
            pivot = matrix.pivot(index='station', columns='save', values='flip_count').fillna(0).astype(int)
            
            # Add totals
            pivot['TOTAL'] = pivot.sum(axis=1)
            pivot.loc['TOTAL'] = pivot.sum(axis=0)
            
            self.df_station_save_matrix = pivot
    
    def analyze_message_types_by_station(self):
        """
        Analyze which message types are causing flips at each station
        """
        if self.df_flips is not None and not self.df_flips.empty:
            msg_station = self.df_flips.groupby(['station', 'msg_type']).agg({
                'filename': 'count',
                'header_issue': 'sum',
                'time_diff_ms': ['mean', 'min', 'max']
            }).reset_index()
            
            msg_station.columns = ['station', 'msg_type', 'flip_count', 'header_issues', 
                                  'avg_time_diff_ms', 'min_time_diff_ms', 'max_time_diff_ms']
            
            msg_station = msg_station.sort_values(['station', 'flip_count'], ascending=[True, False])
            self.df_message_type_station = msg_station
    
    def analyze_flip_patterns(self):
        """
        Analyze patterns in bus flips
        """
        if self.df_flips is not None and not self.df_flips.empty:
            # Analyze transition patterns
            patterns = self.df_flips.groupby(['station', 'bus_transition', 'msg_type']).size().reset_index(name='count')
            patterns = patterns.sort_values('count', ascending=False)
            self.df_flip_patterns = patterns
    
    def create_time_series_analysis(self):
        """
        Create time series data for trend analysis
        """
        if self.df_flips is not None and not self.df_flips.empty:
            # Group flips by hour intervals
            df_ts = self.df_flips.copy()
            df_ts['hour'] = (df_ts['timestamp'] // 3600) * 3600  # Round to hour
            
            ts_data = df_ts.groupby(['station', 'hour']).size().reset_index(name='flips_in_hour')
            ts_data['datetime'] = pd.to_datetime(ts_data['hour'], unit='s')
            
            self.df_time_series = ts_data
    
    def run_analysis(self):
        """
        Run the complete analysis on all CSV files
        """
        if not self.csv_folder.exists():
            print(f"ERROR: CSV folder '{self.csv_folder}' does not exist!")
            return []
            
        csv_files = list(self.csv_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return []
            
        print(f"Found {len(csv_files)} CSV files to process")
        print("-" * 50)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Processing {csv_file.name}...")
            result = self.process_csv(csv_file)
            if result:
                self.all_files_data.append(result)
        
        self.create_summary_dataframes()
        
        # Perform hierarchical aggregations
        print("\nPerforming hierarchical analysis...")
        self.aggregate_station_data()
        self.aggregate_save_data()
        self.create_station_save_matrix()
        self.analyze_message_types_by_station()
        self.analyze_flip_patterns()
        self.create_time_series_analysis()
        
        return self.all_files_data
    
    def create_summary_dataframes(self):
        """
        Create summary dataframes for analysis
        """
        if self.all_files_data:
            self.df_summary = pd.DataFrame(self.all_files_data)
            self.df_summary['flip_rate'] = (self.df_summary['bus_flips'] / self.df_summary['total_rows'] * 1000).round(2)
        
        if self.bus_flip_issues:
            self.df_flips = pd.DataFrame(self.bus_flip_issues)
        
        if self.data_changes:
            self.df_changes = pd.DataFrame(self.data_changes)
        
        if self.header_issues:
            self.df_headers = pd.DataFrame(self.header_issues)
        
        if self.file_message_summary:
            self.df_file_messages = pd.DataFrame(self.file_message_summary)
        
        if self.detailed_flip_data:
            self.df_detailed_flips = pd.DataFrame(self.detailed_flip_data)
    
    def export_to_excel(self):
        """
        Export all dataframes to Excel files with enhanced organization
        """
        excel_path = self.output_folder / "bus_monitor_analysis_enhanced.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1
            })
            
            severity_high = workbook.add_format({'bg_color': '#ffcccc'})
            severity_medium = workbook.add_format({'bg_color': '#ffffcc'})
            severity_low = workbook.add_format({'bg_color': '#ccffcc'})
            
            # 1. Executive Summary (Station Level)
            if self.df_station_summary is not None and len(self.df_station_summary) > 0:
                self.df_station_summary.to_excel(writer, sheet_name='1_Station_Summary', index=False)
                worksheet = writer.sheets['1_Station_Summary']
                
                # Apply conditional formatting to severity score
                worksheet.conditional_format('L2:L100', {
                    'type': '3_color_scale',
                    'min_color': '#00FF00',
                    'mid_color': '#FFFF00',
                    'max_color': '#FF0000'
                })
                print(f"  Exported Station Summary: {len(self.df_station_summary)} stations")
            
            # 2. Save Level Summary
            if self.df_save_summary is not None and len(self.df_save_summary) > 0:
                self.df_save_summary.to_excel(writer, sheet_name='2_Save_Summary', index=False)
                worksheet = writer.sheets['2_Save_Summary']
                
                # Apply conditional formatting
                worksheet.conditional_format('M2:M1000', {
                    'type': '3_color_scale',
                    'min_color': '#00FF00',
                    'mid_color': '#FFFF00',
                    'max_color': '#FF0000'
                })
                print(f"  Exported Save Summary: {len(self.df_save_summary)} station-save combinations")
            
            # 3. Station-Save Matrix
            if self.df_station_save_matrix is not None and len(self.df_station_save_matrix) > 0:
                self.df_station_save_matrix.to_excel(writer, sheet_name='3_Station_Save_Matrix')
                worksheet = writer.sheets['3_Station_Save_Matrix']
                
                # Apply heatmap formatting
                max_val = self.df_station_save_matrix.max().max()
                for row in range(1, len(self.df_station_save_matrix) + 1):
                    for col in range(1, len(self.df_station_save_matrix.columns) + 1):
                        worksheet.conditional_format(row, col, row, col, {
                            'type': '2_color_scale',
                            'min_color': '#FFFFFF',
                            'max_color': '#FF6B6B'
                        })
                print(f"  Exported Station-Save Matrix")
            
            # 4. Message Type Analysis by Station
            if self.df_message_type_station is not None and len(self.df_message_type_station) > 0:
                self.df_message_type_station.to_excel(writer, sheet_name='4_MsgType_by_Station', index=False)
                print(f"  Exported Message Type by Station: {len(self.df_message_type_station)} rows")
            
            # 5. Flip Patterns
            if self.df_flip_patterns is not None and len(self.df_flip_patterns) > 0:
                self.df_flip_patterns.to_excel(writer, sheet_name='5_Flip_Patterns', index=False)
                print(f"  Exported Flip Patterns: {len(self.df_flip_patterns)} patterns")
            
            # 6. File Level Summary
            if self.df_summary is not None and len(self.df_summary) > 0:
                self.df_summary.to_excel(writer, sheet_name='6_File_Summary', index=False)
                print(f"  Exported File Summary: {len(self.df_summary)} files")
            
            # 7. Detailed Flip Records
            if self.df_flips is not None and len(self.df_flips) > 0:
                self.df_flips.to_excel(writer, sheet_name='7_Flip_Details', index=False)
                print(f"  Exported Flip Details: {len(self.df_flips)} flips")
            
            # 8. Data Changes
            if self.df_changes is not None and len(self.df_changes) > 0:
                # Limit to first 10000 rows if too many
                if len(self.df_changes) > 10000:
                    self.df_changes.iloc[:10000].to_excel(writer, sheet_name='8_Data_Changes', index=False)
                    print(f"  Exported Data Changes: 10000 rows (truncated from {len(self.df_changes)})")
                else:
                    self.df_changes.to_excel(writer, sheet_name='8_Data_Changes', index=False)
                    print(f"  Exported Data Changes: {len(self.df_changes)} rows")
            
            # 9. Header Validation Issues
            if self.df_headers is not None and len(self.df_headers) > 0:
                self.df_headers.to_excel(writer, sheet_name='9_Header_Issues', index=False)
                print(f"  Exported Header Issues: {len(self.df_headers)} issues")
            
            # 10. Time Series Data
            if self.df_time_series is not None and len(self.df_time_series) > 0:
                self.df_time_series.to_excel(writer, sheet_name='10_Time_Series', index=False)
                print(f"  Exported Time Series Data: {len(self.df_time_series)} data points")
        
        print(f"\nExcel file saved to: {excel_path.absolute()}")
        return excel_path
    
    def create_enhanced_html_report(self):
        """
        Create an enhanced HTML report with drill-down capabilities
        """
        report_path = self.output_folder / "index.html"
        
        # Calculate summary statistics
        total_stations = len(self.df_station_summary) if self.df_station_summary is not None else 0
        total_saves = len(self.df_save_summary) if self.df_save_summary is not None else 0
        total_files = len(self.df_summary) if self.df_summary is not None else 0
        total_flips = len(self.df_flips) if self.df_flips is not None else 0
        
        # Get top problematic stations
        top_stations_html = ""
        if self.df_station_summary is not None and len(self.df_station_summary) > 0:
            top_5 = self.df_station_summary.nlargest(5, 'severity_score')
            for _, row in top_5.iterrows():
                color = '#ff4444' if row['severity_score'] > 10 else '#ff9944' if row['severity_score'] > 5 else '#44ff44'
                top_stations_html += f"""
                <div style="background: {color}20; padding: 10px; margin: 5px 0; border-left: 4px solid {color};">
                    <strong>{row['station']}</strong><br>
                    Severity: {row['severity_score']:.1f} | Flips: {row['total_flips']} | 
                    Rate: {row['flip_rate_per_1000']:.2f}/1000 | Files: {row['file_count']}
                </div>
                """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Bus Monitor Analysis Dashboard</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    padding: 30px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 15px;
                    margin-bottom: 30px;
                }}
                .hierarchy {{
                    display: flex;
                    justify-content: space-around;
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                .level {{
                    text-align: center;
                    flex: 1;
                    padding: 20px;
                }}
                .level-number {{
                    font-size: 48px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .level-label {{
                    color: #666;
                    margin-top: 10px;
                    font-size: 18px;
                }}
                .arrow {{
                    display: flex;
                    align-items: center;
                    font-size: 30px;
                    color: #667eea;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                    transition: transform 0.3s;
                }}
                .stat-card:hover {{
                    transform: translateY(-5px);
                }}
                .stat-value {{
                    font-size: 36px;
                    font-weight: bold;
                }}
                .stat-label {{
                    margin-top: 10px;
                    opacity: 0.9;
                }}
                .problem-stations {{
                    background: #fff3cd;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 30px 0;
                }}
                .download-section {{
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 10px;
                    margin: 30px 0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 24px;
                    margin: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    transition: all 0.3s;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 7px 20px rgba(0,0,0,0.3);
                }}
                .btn-excel {{
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                }}
                .info-box {{
                    background: #e8f4fd;
                    border-left: 4px solid #2196F3;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 14px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚌 Enhanced Bus Monitor Analysis Dashboard</h1>
                
                <div class="hierarchy">
                    <div class="level">
                        <div class="level-number">{total_stations}</div>
                        <div class="level-label">Stations</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="level">
                        <div class="level-number">{total_saves}</div>
                        <div class="level-label">Station-Save Combos</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="level">
                        <div class="level-number">{total_files}</div>
                        <div class="level-label">Files Analyzed</div>
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_flips:,}</div>
                        <div class="stat-label">Total Bus Flips Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(self.df_headers) if self.df_headers is not None else 0}</div>
                        <div class="stat-label">Header Validation Issues</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(self.df_changes) if self.df_changes is not None else 0:,}</div>
                        <div class="stat-label">Data Word Changes</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{self.df_file_messages['msg_type'].nunique() if self.df_file_messages is not None else 0}</div>
                        <div class="stat-label">Unique Message Types</div>
                    </div>
                </div>
                
                <div class="problem-stations">
                    <h2>⚠️ Top Problem Stations (by Severity Score)</h2>
                    {top_stations_html if top_stations_html else "<p>No stations with issues detected</p>"}
                </div>
                
                <div class="download-section">
                    <h2>📊 Download Analysis Results</h2>
                    
                    <h3>Complete Analysis Package</h3>
                    <a href="bus_monitor_analysis_enhanced.xlsx" class="btn btn-excel">
                        📈 Download Enhanced Excel Report (All Levels)
                    </a>
                    
                    <h3>Individual Analysis Levels</h3>
                    <p>Station Level Analysis:</p>
                    <a href="station_summary.csv" class="btn">Station Summary</a>
                    <a href="station_save_matrix.csv" class="btn">Station-Save Matrix</a>
                    <a href="message_types_by_station.csv" class="btn">Message Types by Station</a>
                    
                    <p>Save Level Analysis:</p>
                    <a href="save_summary.csv" class="btn">Save Summary</a>
                    <a href="flip_patterns.csv" class="btn">Flip Patterns</a>
                    
                    <p>File Level Analysis:</p>
                    <a href="file_summary.csv" class="btn">File Summary</a>
                    <a href="flip_details.csv" class="btn">Detailed Flips</a>
                    <a href="data_changes.csv" class="btn">Data Changes</a>
                </div>
                
                <div class="info-box">
                    <h3>📖 How to Use This Dashboard</h3>
                    <ul>
                        <li><strong>Station Level:</strong> Start here to identify which stations have the most severe bus flip issues</li>
                        <li><strong>Save Level:</strong> Drill down to specific saves within problematic stations</li>
                        <li><strong>File Level:</strong> Examine individual files for detailed flip analysis</li>
                        <li><strong>Severity Score:</strong> Combines flip rate, bus balance, and frequency to identify problem areas</li>
                        <li><strong>Excel Report:</strong> Contains all levels of analysis in organized tabs for easy navigation</li>
                    </ul>
                </div>
                
                <div class="timestamp">
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nEnhanced HTML dashboard saved to: {report_path.absolute()}")
        return report_path
    
    def export_hierarchical_csvs(self):
        """
        Export hierarchical CSV files for each analysis level
        """
        csv_files = []
        
        # Station level
        if self.df_station_summary is not None and len(self.df_station_summary) > 0:
            path = self.output_folder / "station_summary.csv"
            self.df_station_summary.to_csv(path, index=False)
            csv_files.append(path)
        
        # Save level
        if self.df_save_summary is not None and len(self.df_save_summary) > 0:
            path = self.output_folder / "save_summary.csv"
            self.df_save_summary.to_csv(path, index=False)
            csv_files.append(path)
        
        # Station-Save Matrix
        if self.df_station_save_matrix is not None and len(self.df_station_save_matrix) > 0:
            path = self.output_folder / "station_save_matrix.csv"
            self.df_station_save_matrix.to_csv(path)
            csv_files.append(path)
        
        # Message types by station
        if self.df_message_type_station is not None and len(self.df_message_type_station) > 0:
            path = self.output_folder / "message_types_by_station.csv"
            self.df_message_type_station.to_csv(path, index=False)
            csv_files.append(path)
        
        # Flip patterns
        if self.df_flip_patterns is not None and len(self.df_flip_patterns) > 0:
            path = self.output_folder / "flip_patterns.csv"
            self.df_flip_patterns.to_csv(path, index=False)
            csv_files.append(path)
        
        # Original detailed files
        if self.df_summary is not None and len(self.df_summary) > 0:
            path = self.output_folder / "file_summary.csv"
            self.df_summary.to_csv(path, index=False)
            csv_files.append(path)
        
        if self.df_flips is not None and len(self.df_flips) > 0:
            path = self.output_folder / "flip_details.csv"
            self.df_flips.to_csv(path, index=False)
            csv_files.append(path)
        
        if self.df_changes is not None and len(self.df_changes) > 0:
            path = self.output_folder / "data_changes.csv"
            self.df_changes.to_csv(path, index=False)
            csv_files.append(path)
        
        return csv_files
    
    def print_enhanced_summary(self):
        """
        Print enhanced analysis summary with hierarchical insights
        """
        print("\n" + "="*70)
        print("ENHANCED BUS MONITOR ANALYSIS SUMMARY")
        print("="*70)
        
        # Station Level Summary
        if self.df_station_summary is not None and len(self.df_station_summary) > 0:
            print("\n📍 STATION LEVEL ANALYSIS")
            print("-"*40)
            print(f"Total Stations Analyzed: {len(self.df_station_summary)}")
            
            # Top 3 problem stations
            top_3 = self.df_station_summary.nlargest(3, 'severity_score')
            print("\nTop 3 Problem Stations (by severity):")
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"  {idx}. {row['station']}")
                print(f"     - Severity Score: {row['severity_score']:.2f}")
                print(f"     - Total Flips: {row['total_flips']}")
                print(f"     - Flip Rate: {row['flip_rate_per_1000']:.2f} per 1000 messages")
                print(f"     - Files: {row['file_count']}, Saves: {row['save_count']}")
        
        # Save Level Summary
        if self.df_save_summary is not None and len(self.df_save_summary) > 0:
            print("\n💾 SAVE LEVEL ANALYSIS")
            print("-"*40)
            print(f"Total Station-Save Combinations: {len(self.df_save_summary)}")
            
            # Top 3 problem saves
            top_3_saves = self.df_save_summary.nlargest(3, 'severity_score')
            print("\nTop 3 Problem Saves:")
            for idx, (_, row) in enumerate(top_3_saves.iterrows(), 1):
                print(f"  {idx}. {row['station']} - {row['save']}")
                print(f"     - Severity Score: {row['severity_score']:.2f}")
                print(f"     - Total Flips: {row['total_flips']}")
                print(f"     - Files: {row['file_count']}")
        
        # Message Type Analysis
        if self.df_message_type_station is not None and len(self.df_message_type_station) > 0:
            print("\n📨 MESSAGE TYPE ANALYSIS")
            print("-"*40)
            top_msg_types = self.df_message_type_station.groupby('msg_type')['flip_count'].sum().nlargest(5)
            print("Top 5 Message Types Causing Flips:")
            for msg_type, count in top_msg_types.items():
                print(f"  - {msg_type}: {count} flips")
        
        # Overall Statistics
        print("\n📊 OVERALL STATISTICS")
        print("-"*40)
        if self.df_summary is not None:
            print(f"Total Files Processed: {len(self.df_summary)}")
            print(f"Total Rows Analyzed: {self.df_summary['total_rows'].sum():,}")
        
        if self.df_flips is not None:
            print(f"Total Bus Flips Detected: {len(self.df_flips)}")
        
        if self.df_headers is not None:
            print(f"Header Validation Issues: {len(self.df_headers)}")
        
        if self.df_changes is not None:
            print(f"Total Data Word Changes: {len(self.df_changes)}")
        
        print("\n" + "="*70)


def main():
    """
    Main execution function
    """
    print("Starting Enhanced Bus Monitor Analysis")
    print("-" * 60)
    
    analyzer = EnhancedBusMonitorDashboard()
    
    results = analyzer.run_analysis()
    
    if not results:
        print("\nNo data was processed. Please check:")
        print(f"  1. The CSV folder path is correct: {analyzer.csv_folder}")
        print("  2. CSV files exist in that folder")
        print("  3. CSV files have the correct format")
        return
    
    analyzer.print_enhanced_summary()
    
    print("\nExporting results...")
    print("-" * 60)
    
    # Export to Excel with all levels
    excel_path = analyzer.export_to_excel()
    
    # Export hierarchical CSVs
    print("\nExporting hierarchical CSV files...")
    csv_files = analyzer.export_hierarchical_csvs()
    
    # Create enhanced HTML dashboard
    html_path = analyzer.create_enhanced_html_report()
    
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {analyzer.output_folder.absolute()}")
    print("\nAnalysis Levels Available:")
    print("  1. STATION LEVEL - Identify problematic stations")
    print("  2. SAVE LEVEL - Drill down to specific saves")
    print("  3. FILE LEVEL - Examine individual file issues")
    print("\nFiles Generated:")
    print(f"  📊 Excel Report: {excel_path.name} (all levels in tabs)")
    print(f"  🌐 HTML Dashboard: {html_path.name} (interactive overview)")
    print(f"  📁 CSV Files: {len(csv_files)} files for detailed analysis")
    print("\nUse the HTML dashboard for quick insights and navigation!")


if __name__ == "__main__":
    main()
