#!/usr/bin/env python3
"""
Bus Monitor Analysis Dashboard - Spreadsheet-focused Version
Tracks bus flips with matching decoded_description values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Set
warnings.filterwarnings('ignore')


class BusMonitorDashboard:
    def __init__(self):
        """
        Initialize the Bus Monitor Dashboard
        """
        # CONFIGURATION - Change these paths as needed
        self.csv_folder = Path("./csv_data")  # <-- UPDATE THIS PATH
        self.lookup_csv_path = Path("./message_lookup.csv")  # <-- PATH TO LOOKUP CSV
        
        # Analysis results storage
        self.bus_flip_issues = []
        self.data_changes = []
        self.flip_statistics = defaultdict(lambda: defaultdict(int))
        self.message_type_stats = defaultdict(lambda: defaultdict(set))
        self.all_files_data = []
        self.header_issues = []
        self.file_message_summary = []  # Track messages per file
        self.detailed_flip_data = []  # Store complete row data for flips
        
        # DataFrames for analysis
        self.df_summary = None
        self.df_flips = None
        self.df_changes = None
        self.df_headers = None
        self.df_file_messages = None
        self.df_detailed_flips = None
        
        # Load message type to header lookup
        self.message_header_lookup = self.load_message_lookup()
    
    def load_message_lookup(self):
        """
        Load the message type to header lookup table
        Returns a dictionary mapping message_type to list of valid headers
        """
        lookup = defaultdict(list)
        
        if self.lookup_csv_path.exists():
            try:
                df_lookup = pd.read_csv(self.lookup_csv_path)
                print(f"Loaded message lookup from {self.lookup_csv_path}")
                
                # Group by message_type to handle multiple valid headers
                for msg_type, group in df_lookup.groupby('message_type'):
                    lookup[msg_type] = group['header'].tolist()
                
                print(f"  Loaded {len(lookup)} message types with header mappings")
                
            except Exception as e:
                print(f"Warning: Could not load message lookup: {e}")
        else:
            print(f"Note: No message lookup file found at {self.lookup_csv_path}")
            print("  Header validation will be skipped")
        
        return lookup
    
    def validate_header(self, msg_type: str, actual_header: str) -> Tuple[bool, List[str]]:
        """
        Validate if the actual header (data01) matches expected headers for message type
        Returns: (is_valid, list_of_expected_headers)
        """
        if not self.message_header_lookup or msg_type is None:
            return True, []
        
        expected_headers = self.message_header_lookup.get(msg_type, [])
        
        if not expected_headers:
            return True, []
        
        # Check if actual header matches any of the expected headers
        actual_clean = str(actual_header).strip().upper() if pd.notna(actual_header) else ""
        expected_clean = [str(h).strip().upper() for h in expected_headers]
        
        is_valid = actual_clean in expected_clean
        
        return is_valid, expected_headers
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename to extract unit_id, station, save, and station_num
        Format: unit_id_station_save_rt01
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
    
    def extract_message_type(self, decoded_desc: str) -> Tuple[str, int]:
        """
        Extract message type and expected data words from decoded description
        Format: "(1-[190R]-1)" or "(1-[190R-1]-1)"
        Returns: (message_type, num_data_words)
        """
        if pd.isna(decoded_desc):
            return None, 0
            
        # Pattern to match message types with or without hyphens
        pattern = r'\((\d+)-\[([^\]]+)\]-(\d+)\)'
        match = re.search(pattern, str(decoded_desc))
        
        if match:
            start_num = match.group(1)
            msg_type = match.group(2)
            end_num = int(match.group(3))
            return msg_type, end_num
        
        return None, 0
    
    def detect_bus_flips(self, df: pd.DataFrame, file_info: Dict) -> List[Dict]:
        """
        Detect rapid bus flips (A to B or B to A within 100ms)
        ONLY tracks flips with matching decoded_description values
        """
        flips = []
        
        # Ensure required columns exist
        if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
            print(f"Warning: Missing required columns in {file_info['filename']}")
            return flips
            
        # Prepare dataframe
        df = df.copy()
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            return flips
        
        # Vectorized operations using shift
        df['prev_bus'] = df['bus'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        df['prev_decoded'] = df['decoded_description'].shift(1)
        
        # Calculate time difference in milliseconds
        df['time_diff_ms'] = (df['timestamp'] - df['prev_timestamp']) * 1000
        
        # Create boolean mask for bus changes within 100ms WITH SAME decoded_description
        mask = (
            (df['bus'] != df['prev_bus']) & 
            (df['time_diff_ms'] < 100) & 
            (df['time_diff_ms'].notna()) &
            (df['decoded_description'] == df['prev_decoded'])  # ONLY matching descriptions
        )
        
        # Get indices where flips occur
        flip_indices = df[mask].index.tolist()
        
        # Get all column names for detailed tracking
        all_columns = df.columns.tolist()
        
        # Process only the rows with flips
        for idx in flip_indices:
            curr_row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            # Extract message type
            msg_type, num_words = self.extract_message_type(curr_row.get('decoded_description'))
            
            # Validate header if data01 exists
            header_issue = None
            if 'data01' in df.columns and msg_type:
                is_valid_prev, expected = self.validate_header(msg_type, prev_row.get('data01'))
                is_valid_curr, _ = self.validate_header(msg_type, curr_row.get('data01'))
                
                if not is_valid_prev or not is_valid_curr:
                    header_issue = {
                        'expected': expected,
                        'actual_prev': prev_row.get('data01'),
                        'actual_curr': curr_row.get('data01')
                    }
                    
                    # Track header issue
                    self.header_issues.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'filename': file_info['filename'],
                        'timestamp': prev_row['timestamp'],
                        'msg_type': msg_type,
                        'actual_header_busA': prev_row.get('data01') if prev_row['bus'] == 'A' else curr_row.get('data01'),
                        'actual_header_busB': prev_row.get('data01') if prev_row['bus'] == 'B' else curr_row.get('data01'),
                        'expected_headers': ', '.join(map(str, expected))
                    })
            
            # Create flip record
            flip_info = {
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'filename': file_info['filename'],
                'timestamp_prev': prev_row['timestamp'],
                'timestamp_curr': curr_row['timestamp'],
                'time_diff_ms': round(curr_row['time_diff_ms'], 2),
                'bus_transition': f"{prev_row['bus']} to {curr_row['bus']}",
                'bus_prev': prev_row['bus'],
                'bus_curr': curr_row['bus'],
                'msg_type': msg_type,
                'decoded_description': curr_row['decoded_description'],
                'data01_prev': prev_row.get('data01', ''),
                'data01_curr': curr_row.get('data01', ''),
                'header_issue': header_issue is not None,
                'row_index_prev': idx - 1,
                'row_index_curr': idx
            }
            
            # Track data changes
            data_changes = self.compare_data_words(prev_row, curr_row)
            flip_info['num_data_changes'] = len(data_changes)
            
            # Store detailed changes
            for col, change in data_changes.items():
                self.data_changes.append({
                    'unit_id': file_info['unit_id'],
                    'station': file_info['station'],
                    'save': file_info['save'],
                    'filename': file_info['filename'],
                    'timestamp': prev_row['timestamp'],
                    'msg_type': msg_type,
                    'decoded_description': curr_row['decoded_description'],
                    'data_column': col,
                    'value_busA': change['before'] if prev_row['bus'] == 'A' else change['after'],
                    'value_busB': change['before'] if prev_row['bus'] == 'B' else change['after'],
                    'bus_prev': prev_row['bus'],
                    'bus_curr': curr_row['bus']
                })
            
            # Store complete row data for detailed view
            detailed_flip = {
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'filename': file_info['filename'],
                'flip_timestamp': prev_row['timestamp'],
                'bus_transition': f"{prev_row['bus']} to {curr_row['bus']}",
                'decoded_description': curr_row['decoded_description'],
                'msg_type': msg_type
            }
            
            # Add all data columns from both rows
            for col in all_columns:
                if col.startswith('data') or col in ['timestamp', 'bus', 'decoded_description']:
                    detailed_flip[f'{col}_busA'] = prev_row[col] if prev_row['bus'] == 'A' else curr_row[col]
                    detailed_flip[f'{col}_busB'] = prev_row[col] if prev_row['bus'] == 'B' else curr_row[col]
            
            self.detailed_flip_data.append(detailed_flip)
            
            flips.append(flip_info)
            
            # Update statistics
            key = (file_info['unit_id'], file_info['station'], file_info['save'], msg_type or 'unknown')
            self.flip_statistics[key]['flip_count'] += 1
            if header_issue:
                self.flip_statistics[key]['header_issues'] += 1
        
        return flips
    
    def compare_data_words(self, row1: pd.Series, row2: pd.Series) -> Dict[str, Dict]:
        """
        Compare data word columns between two rows to identify changes
        """
        changes = {}
        
        # Find all data columns
        data_cols = [col for col in row1.index if col.startswith('data') and 
                    len(col) > 4 and col[4:].replace('0', '').isdigit()]
        
        for col in data_cols:
            if col in row2.index:
                val1, val2 = row1[col], row2[col]
                
                # Skip if both are NaN
                if pd.isna(val1) and pd.isna(val2):
                    continue
                    
                # Compare as strings
                if str(val1) != str(val2):
                    changes[col] = {
                        'before': val1,
                        'after': val2,
                        'bus_before': row1['bus'],
                        'bus_after': row2['bus']
                    }
        
        return changes
    
    def analyze_file_messages(self, df: pd.DataFrame, file_info: Dict):
        """
        Analyze all messages in a file for summary
        """
        if 'decoded_description' not in df.columns:
            return
        
        # Get unique messages in this file
        unique_messages = df['decoded_description'].dropna().unique()
        
        for decoded_desc in unique_messages:
            msg_type, num_words = self.extract_message_type(decoded_desc)
            
            # Get statistics for this message
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
    
    def process_csv(self, csv_path: Path) -> Dict:
        """
        Process a single CSV file
        """
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            print(f"Warning: Could not parse filename {csv_path.name}")
            return None
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Check required columns
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {csv_path.name}")
                return None
            
            # Detect bus flips
            flips = self.detect_bus_flips(df, file_info)
            if flips:
                self.bus_flip_issues.extend(flips)
            
            # Analyze messages in file
            self.analyze_file_messages(df, file_info)
            
            # Calculate statistics
            bus_counts = df['bus'].value_counts()
            
            return {
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'total_rows': len(df),
                'bus_flips': len(flips),
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0),
                'unique_messages': df['decoded_description'].nunique() if 'decoded_description' in df.columns else 0
            }
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            return None
    
    def run_analysis(self):
        """
        Run the complete analysis on all CSV files
        """
        if not self.csv_folder.exists():
            print(f"ERROR: CSV folder '{self.csv_folder}' does not exist!")
            print(f"Please update the csv_folder path in the __init__ method")
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
        
        # Create summary dataframes
        self.create_summary_dataframes()
        
        return self.all_files_data
    
    def create_summary_dataframes(self):
        """
        Create summary dataframes for visualization
        """
        # File summary
        if self.all_files_data:
            self.df_summary = pd.DataFrame(self.all_files_data)
            self.df_summary['flip_rate'] = (self.df_summary['bus_flips'] / self.df_summary['total_rows'] * 1000).round(2)
        
        # Flip issues
        if self.bus_flip_issues:
            self.df_flips = pd.DataFrame(self.bus_flip_issues)
        
        # Data changes
        if self.data_changes:
            self.df_changes = pd.DataFrame(self.data_changes)
        
        # Header issues
        if self.header_issues:
            self.df_headers = pd.DataFrame(self.header_issues)
        
        # File message summary
        if self.file_message_summary:
            self.df_file_messages = pd.DataFrame(self.file_message_summary)
        
        # Detailed flip data
        if self.detailed_flip_data:
            self.df_detailed_flips = pd.DataFrame(self.detailed_flip_data)
    
    def create_file_summary_table(self):
        """
        Create spreadsheet view of files with bus flips
        """
        if self.df_summary is None or len(self.df_summary) == 0:
            return None
        
        # Sort by flip count
        table_data = self.df_summary.sort_values('bus_flips', ascending=False)
        
        # Prepare display values as lists
        display_values = [
            table_data['filename'].tolist(),
            table_data['unit_id'].tolist(),
            table_data['station'].tolist(),
            table_data['save'].tolist(),
            table_data['total_rows'].tolist(),
            table_data['bus_flips'].tolist(),
            table_data['flip_rate'].tolist(),
            table_data['bus_a_count'].tolist(),
            table_data['bus_b_count'].tolist(),
            table_data['unique_messages'].tolist()
        ]
        
        fig = go.Figure(data=[go.Table(
            columnwidth=[150, 80, 80, 80, 100, 80, 100, 100, 100, 120],
            header=dict(
                values=['Filename', 'Unit ID', 'Station', 'Save', 
                       'Total Rows', 'Bus Flips', 'Flip Rate/1k', 
                       'Bus A Count', 'Bus B Count', 'Unique Messages'],
                fill_color='lightgray',
                align='left',
                font=dict(size=11, color='black'),
                height=30
            ),
            cells=dict(
                values=display_values,
                fill_color='white',
                align='left',
                font=dict(size=10),
                height=25
            )
        )])
        
        fig.update_layout(
            title="File Summary - All Processed Files",
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
    
    def create_message_summary_table(self):
        """
        Create spreadsheet view of message types in files
        """
        if self.df_file_messages is None or len(self.df_file_messages) == 0:
            return None
        
        # Sort by filename and message type
        table_data = self.df_file_messages.sort_values(['filename', 'msg_type'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Filename', 'Unit ID', 'Station', 'Save',
                       'Message Type', 'Decoded Description', 
                       'Total Count', 'Bus A Count', 'Bus B Count', 'Has Flips'],
                fill_color='lightgray',
                align='left',
                font=dict(size=11, color='black')
            ),
            cells=dict(
                values=[
                    table_data['filename'],
                    table_data['unit_id'],
                    table_data['station'],
                    table_data['save'],
                    table_data['msg_type'],
                    table_data['decoded_description'],
                    table_data['total_count'],
                    table_data['bus_a_count'],
                    table_data['bus_b_count'],
                    table_data['has_flips'].apply(lambda x: 'Yes' if x else 'No')
                ],
                fill_color='white',
                align='left',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title="Message Type Summary by File",
            height=600
        )
        
        return fig
    
    def create_flip_details_table(self):
        """
        Create detailed spreadsheet of all bus flips
        """
        if self.df_flips is None or len(self.df_flips) == 0:
            return None
        
        # Sort by filename and timestamp
        table_data = self.df_flips.sort_values(['filename', 'timestamp_prev'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Filename', 'Unit ID', 'Station', 'Save',
                       'Timestamp', 'Time Diff (ms)', 'Bus Transition',
                       'Message Type', 'Decoded Description',
                       'Data01 Prev', 'Data01 Curr', 'Data Changes', 'Header Issue'],
                fill_color='lightgray',
                align='left',
                font=dict(size=11, color='black')
            ),
            cells=dict(
                values=[
                    table_data['filename'],
                    table_data['unit_id'],
                    table_data['station'],
                    table_data['save'],
                    table_data['timestamp_prev'].round(4),
                    table_data['time_diff_ms'],
                    table_data['bus_transition'],
                    table_data['msg_type'],
                    table_data['decoded_description'],
                    table_data['data01_prev'],
                    table_data['data01_curr'],
                    table_data['num_data_changes'],
                    table_data['header_issue'].apply(lambda x: 'Yes' if x else 'No')
                ],
                fill_color='white',
                align='left',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title="Bus Flip Details - All Detected Flips",
            height=600
        )
        
        return fig
    
    def create_data_changes_table(self):
        """
        Create spreadsheet of data word changes during flips
        """
        if self.df_changes is None or len(self.df_changes) == 0:
            return None
        
        # Sort by filename and timestamp
        table_data = self.df_changes.sort_values(['filename', 'timestamp'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Filename', 'Unit ID', 'Station', 'Save',
                       'Timestamp', 'Message Type', 'Decoded Description',
                       'Data Column', 'Value Bus A', 'Value Bus B',
                       'Bus Transition'],
                fill_color='lightgray',
                align='left',
                font=dict(size=11, color='black')
            ),
            cells=dict(
                values=[
                    table_data['filename'],
                    table_data['unit_id'],
                    table_data['station'],
                    table_data['save'],
                    table_data['timestamp'].round(4),
                    table_data['msg_type'],
                    table_data['decoded_description'],
                    table_data['data_column'],
                    table_data['value_busA'],
                    table_data['value_busB'],
                    table_data.apply(lambda x: f"{x['bus_prev']} to {x['bus_curr']}", axis=1)
                ],
                fill_color='white',
                align='left',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title="Data Word Changes During Bus Flips",
            height=600
        )
        
        return fig
    
    def create_complete_flip_data_table(self):
        """
        Create comprehensive spreadsheet with all columns for flipped messages
        """
        if self.df_detailed_flips is None or len(self.df_detailed_flips) == 0:
            return None
        
        # Get all columns
        all_cols = self.df_detailed_flips.columns.tolist()
        
        # Organize columns logically
        id_cols = ['filename', 'unit_id', 'station', 'save', 'flip_timestamp', 
                   'bus_transition', 'msg_type', 'decoded_description']
        data_cols = [col for col in all_cols if col not in id_cols]
        
        ordered_cols = id_cols + sorted(data_cols)
        
        # Create table with horizontal scrolling for many columns
        table_data = self.df_detailed_flips[ordered_cols].sort_values(['filename', 'flip_timestamp'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=ordered_cols,
                fill_color='lightgray',
                align='left',
                font=dict(size=10, color='black')
            ),
            cells=dict(
                values=[table_data[col] for col in ordered_cols],
                fill_color='white',
                align='left',
                font=dict(size=9)
            )
        )])
        
        fig.update_layout(
            title="Complete Flip Data - All Columns",
            height=600
        )
        
        return fig
    
    def create_header_validation_table(self):
        """
        Create spreadsheet of header validation issues
        """
        if self.df_headers is None or len(self.df_headers) == 0:
            return None
        
        table_data = self.df_headers.sort_values(['filename', 'timestamp'])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Filename', 'Unit ID', 'Station', 'Save',
                       'Timestamp', 'Message Type', 
                       'Header Bus A', 'Header Bus B', 'Expected Headers'],
                fill_color='lightgray',
                align='left',
                font=dict(size=11, color='black')
            ),
            cells=dict(
                values=[
                    table_data['filename'],
                    table_data['unit_id'],
                    table_data['station'],
                    table_data['save'],
                    table_data['timestamp'].round(4),
                    table_data['msg_type'],
                    table_data['actual_header_busA'],
                    table_data['actual_header_busB'],
                    table_data['expected_headers']
                ],
                fill_color='white',
                align='left',
                font=dict(size=10)
            )
        )])
        
        fig.update_layout(
            title="Header Validation Issues",
            height=600
        )
        
        return fig
    
    def print_summary(self):
        """
        Print analysis summary to console
        """
        print("\n" + "="*60)
        print("BUS MONITOR ANALYSIS SUMMARY")
        print("="*60)
        
        if self.df_summary is not None:
            print(f"\nFiles Processed: {len(self.df_summary)}")
            print(f"Total Rows Analyzed: {self.df_summary['total_rows'].sum():,}")
        
        if self.df_flips is not None and len(self.df_flips) > 0:
            print(f"\nBus Flip Issues Found: {len(self.df_flips)}")
            print(f"  (Only counting flips with matching decoded_description)")
            print(f"Average Time Between Flips: {self.df_flips['time_diff_ms'].mean():.2f} ms")
            print(f"Min Time Between Flips: {self.df_flips['time_diff_ms'].min():.2f} ms")
            print(f"Max Time Between Flips: {self.df_flips['time_diff_ms'].max():.2f} ms")
            
            # Most problematic units
            print(f"\nTop 5 Most Problematic Units:")
            top_units = self.df_flips['unit_id'].value_counts().head(5)
            for unit, count in top_units.items():
                print(f"  - {unit}: {count} flips")
            
            # Most common message types with issues
            print(f"\nTop 5 Message Types with Flips:")
            top_msgs = self.df_flips['msg_type'].value_counts().head(5)
            for msg, count in top_msgs.items():
                print(f"  - {msg}: {count} flips")
        
        if self.df_headers is not None and len(self.df_headers) > 0:
            print(f"\nHeader Validation Issues: {len(self.df_headers)}")
        
        if self.df_changes is not None and len(self.df_changes) > 0:
            print(f"\nTotal Data Word Changes: {len(self.df_changes)}")
            print(f"Unique Data Columns Affected: {self.df_changes['data_column'].nunique()}")
        
        print("\n" + "="*60)
    
    def save_analysis_report(self):
        """
        Save HTML report with spreadsheet-style tables
        """
        report_path = Path("bus_monitor_analysis_report.html")
        
        # Calculate summary statistics
        total_files = len(self.df_summary) if self.df_summary is not None else 0
        total_rows = self.df_summary['total_rows'].sum() if self.df_summary is not None else 0
        total_flips = len(self.df_flips) if self.df_flips is not None else 0
        total_changes = len(self.df_changes) if self.df_changes is not None else 0
        header_issues = len(self.df_headers) if self.df_headers is not None else 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bus Monitor Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: #2c3e50;
                    color: white;
                    padding: 20px;
                }}
                h1 {{
                    margin: 0;
                    font-size: 2em;
                }}
                .subtitle {{
                    opacity: 0.9;
                    margin-top: 10px;
                    font-size: 0.9em;
                }}
                .stats-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    padding: 15px;
                    background: white;
                    margin: 15px;
                    border-radius: 5px;
                }}
                .stat-box {{
                    padding: 10px 20px;
                    background: #ecf0f1;
                    border-radius: 5px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                .tabs {{
                    display: flex;
                    background: white;
                    margin: 0 15px;
                    padding: 0;
                    border-radius: 5px 5px 0 0;
                }}
                .tab {{
                    padding: 12px 20px;
                    cursor: pointer;
                    border: none;
                    background: none;
                    font-size: 14px;
                    color: #7f8c8d;
                    border-bottom: 2px solid transparent;
                }}
                .tab:hover {{
                    color: #2c3e50;
                }}
                .tab.active {{
                    color: #2c3e50;
                    border-bottom-color: #3498db;
                }}
                .tab-content {{
                    display: none;
                    padding: 15px;
                    background: white;
                    margin: 0 15px 15px 15px;
                    border-radius: 0 0 5px 5px;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .table-container {{
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bus Monitor Analysis Report</h1>
                <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                <div class="subtitle">Only tracking bus flips with matching decoded_description values</div>
            </div>
            
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-label">Files Analyzed</div>
                    <div class="stat-value">{total_files}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value">{total_rows:,}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Bus Flips</div>
                    <div class="stat-value">{total_flips}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Data Changes</div>
                    <div class="stat-value">{total_changes}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Header Issues</div>
                    <div class="stat-value">{header_issues}</div>
                </div>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="showTab(event, 'files')">File Summary</button>
                <button class="tab" onclick="showTab(event, 'messages')">Message Types</button>
                <button class="tab" onclick="showTab(event, 'flips')">Flip Details</button>
                <button class="tab" onclick="showTab(event, 'changes')">Data Changes</button>
                <button class="tab" onclick="showTab(event, 'complete')">Complete Flip Data</button>
                <button class="tab" onclick="showTab(event, 'headers')">Header Validation</button>
            </div>
            
            <div id="files" class="tab-content active">
                <div class="table-container">
        """
        
        # Add file summary table
        file_table = self.create_file_summary_table()
        if file_table:
            html_content += file_table.to_html(include_plotlyjs=False, div_id="file_table")
        
        html_content += """
                </div>
            </div>
            
            <div id="messages" class="tab-content">
                <div class="table-container">
        """
        
        # Add message summary table
        msg_table = self.create_message_summary_table()
        if msg_table:
            html_content += msg_table.to_html(include_plotlyjs=False, div_id="msg_table")
        
        html_content += """
                </div>
            </div>
            
            <div id="flips" class="tab-content">
                <div class="table-container">
        """
        
        # Add flip details table
        flip_table = self.create_flip_details_table()
        if flip_table:
            html_content += flip_table.to_html(include_plotlyjs=False, div_id="flip_table")
        
        html_content += """
                </div>
            </div>
            
            <div id="changes" class="tab-content">
                <div class="table-container">
        """
        
        # Add data changes table
        changes_table = self.create_data_changes_table()
        if changes_table:
            html_content += changes_table.to_html(include_plotlyjs=False, div_id="changes_table")
        
        html_content += """
                </div>
            </div>
            
            <div id="complete" class="tab-content">
                <div class="table-container">
        """
        
        # Add complete flip data table
        complete_table = self.create_complete_flip_data_table()
        if complete_table:
            html_content += complete_table.to_html(include_plotlyjs=False, div_id="complete_table")
        
        html_content += """
                </div>
            </div>
            
            <div id="headers" class="tab-content">
                <div class="table-container">
        """
        
        # Add header validation table
        header_table = self.create_header_validation_table()
        if header_table:
            html_content += header_table.to_html(include_plotlyjs=False, div_id="header_table")
        else:
            html_content += "<p>No header validation issues found or no lookup table provided.</p>"
        
        html_content += """
                </div>
            </div>
            
            <script>
                function showTab(evt, tabName) {
                    var i, tabcontent, tabs;
                    
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].classList.remove("active");
                    }
                    
                    tabs = document.getElementsByClassName("tab");
                    for (i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove("active");
                    }
                    
                    document.getElementById(tabName).classList.add("active");
                    evt.currentTarget.classList.add("active");
                }
            </script>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nAnalysis report saved to: {report_path.absolute()}")
        print("Open this file in your browser to view the spreadsheet tables")


def main():
    """
    Main execution function
    """
    print("Starting Bus Monitor Analysis")
    print("-" * 60)
    
    # Create analyzer
    analyzer = BusMonitorDashboard()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if not results:
        print("\nNo data was processed. Please check:")
        print(f"  1. The CSV folder path is correct: {analyzer.csv_folder}")
        print("  2. CSV files exist in that folder")
        print("  3. CSV files have the correct format")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Create and save report
    print("\nGenerating analysis report...")
    analyzer.save_analysis_report()
    
    print("\nAnalysis Complete!")


if __name__ == "__main__":
    main()
