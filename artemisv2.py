#!/usr/bin/env python3
"""
Bus Monitor Analysis Dashboard - Enhanced Version with Header Validation
Interactive dashboard with advanced filtering, timeline analysis, and header validation
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
import plotly.express as px
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
        self.header_issues = []  # Track header mismatches
        
        # DataFrames for analysis
        self.df_summary = None
        self.df_flips = None
        self.df_changes = None
        self.df_headers = None  # DataFrame for header validation issues
        
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
                print(f"✓ Loaded message lookup from {self.lookup_csv_path}")
                
                # Group by message_type to handle multiple valid headers
                for msg_type, group in df_lookup.groupby('message_type'):
                    # Store all valid headers for this message type
                    lookup[msg_type] = group['header'].tolist()
                
                print(f"  → Loaded {len(lookup)} message types with header mappings")
                
            except Exception as e:
                print(f"Warning: Could not load message lookup: {e}")
        else:
            print(f"Note: No message lookup file found at {self.lookup_csv_path}")
            print("  → Header validation will be skipped")
        
        return lookup
    
    def validate_header(self, msg_type: str, actual_header: str) -> Tuple[bool, List[str]]:
        """
        Validate if the actual header (data01) matches expected headers for message type
        Returns: (is_valid, list_of_expected_headers)
        """
        if not self.message_header_lookup or msg_type is None:
            return True, []  # Skip validation if no lookup available
        
        expected_headers = self.message_header_lookup.get(msg_type, [])
        
        if not expected_headers:
            # No expected headers defined for this message type
            return True, []
        
        # Check if actual header matches any of the expected headers
        # Handle case variations and potential formatting differences
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
        Also validates headers against expected values
        Optimized using vectorized operations for better performance
        """
        flips = []
        
        # Ensure required columns exist
        if 'timestamp' not in df.columns:
            print(f"Warning: No timestamp column in {file_info['filename']}")
            return flips
            
        # Prepare dataframe - timestamp is already numeric
        df = df.copy()
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            return flips
        
        # Vectorized operations using shift
        df['prev_bus'] = df['bus'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        
        # Calculate time difference in milliseconds (assuming timestamp is in seconds)
        df['time_diff_ms'] = (df['timestamp'] - df['prev_timestamp']) * 1000
        
        # Create boolean mask for bus changes within 100ms
        mask = (df['bus'] != df['prev_bus']) & (df['time_diff_ms'] < 100) & (df['time_diff_ms'].notna())
        
        # Get indices where flips occur
        flip_indices = df[mask].index.tolist()
        
        # Process only the rows with flips
        for idx in flip_indices:
            curr_row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            # Extract message types
            msg_type_prev, dw_prev = self.extract_message_type(prev_row.get('decoded_description'))
            msg_type_curr, dw_curr = self.extract_message_type(curr_row.get('decoded_description'))
            
            # Validate headers if data01 column exists
            header_issue_prev = None
            header_issue_curr = None
            
            if 'data01' in df.columns:
                # Check previous message header
                if msg_type_prev:
                    is_valid_prev, expected_prev = self.validate_header(msg_type_prev, prev_row.get('data01'))
                    if not is_valid_prev:
                        header_issue_prev = {
                            'actual': prev_row.get('data01'),
                            'expected': expected_prev,
                            'msg_type': msg_type_prev
                        }
                        # Track header issue
                        self.header_issues.append({
                            'unit_id': file_info['unit_id'],
                            'station': file_info['station'],
                            'save': file_info['save'],
                            'timestamp': prev_row['timestamp'],
                            'msg_type': msg_type_prev,
                            'actual_header': prev_row.get('data01'),
                            'expected_headers': ', '.join(map(str, expected_prev)),
                            'bus': prev_row['bus'],
                            'context': 'bus_flip'
                        })
                
                # Check current message header
                if msg_type_curr:
                    is_valid_curr, expected_curr = self.validate_header(msg_type_curr, curr_row.get('data01'))
                    if not is_valid_curr:
                        header_issue_curr = {
                            'actual': curr_row.get('data01'),
                            'expected': expected_curr,
                            'msg_type': msg_type_curr
                        }
                        # Track header issue
                        self.header_issues.append({
                            'unit_id': file_info['unit_id'],
                            'station': file_info['station'],
                            'save': file_info['save'],
                            'timestamp': curr_row['timestamp'],
                            'msg_type': msg_type_curr,
                            'actual_header': curr_row.get('data01'),
                            'expected_headers': ', '.join(map(str, expected_curr)),
                            'bus': curr_row['bus'],
                            'context': 'bus_flip'
                        })
            
            flip_info = {
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'filename': file_info['filename'],
                'timestamp_prev': prev_row['timestamp'],
                'timestamp_curr': curr_row['timestamp'],
                'time_diff_ms': round(curr_row['time_diff_ms'], 2),
                'bus_transition': f"{prev_row['bus']} → {curr_row['bus']}",
                'bus_prev': prev_row['bus'],
                'bus_curr': curr_row['bus'],
                'msg_type_prev': msg_type_prev,
                'msg_type_curr': msg_type_curr,
                'decoded_desc_prev': prev_row.get('decoded_description'),
                'decoded_desc_curr': curr_row.get('decoded_description'),
                'data01_prev': prev_row.get('data01'),
                'data01_curr': curr_row.get('data01'),
                'header_issue_prev': header_issue_prev,
                'header_issue_curr': header_issue_curr,
                'row_index_prev': idx - 1,
                'row_index_curr': idx
            }
            
            # Track data changes
            data_changes = self.compare_data_words(prev_row, curr_row)
            if data_changes:
                flip_info['data_changes'] = data_changes
                flip_info['num_changes'] = len(data_changes)
                
                # Add individual change records
                for col, change in data_changes.items():
                    self.data_changes.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'timestamp': prev_row['timestamp'],
                        'data_column': col,
                        'value_before': change['before'],
                        'value_after': change['after'],
                        'bus_before': change['bus_before'],
                        'bus_after': change['bus_after'],
                        'msg_type': msg_type_prev
                    })
            else:
                flip_info['num_changes'] = 0
            
            flips.append(flip_info)
            
            # Update statistics
            key = (file_info['unit_id'], file_info['station'], 
                  file_info['save'], msg_type_prev or 'unknown')
            self.flip_statistics[key]['flip_count'] += 1
            
            # Track if header issues are involved
            if header_issue_prev or header_issue_curr:
                self.flip_statistics[key]['header_issues'] += 1
        
        # Print performance stats for large files
        if len(df) > 100000:
            print(f"  → Processed {len(df):,} rows, found {len(flips)} flips in {file_info['filename']}")
                    
        return flips
    
    def compare_data_words(self, row1: pd.Series, row2: pd.Series) -> Dict[str, Dict]:
        """
        Compare data word columns between two rows to identify changes
        Optimized for performance using vectorized comparison
        """
        changes = {}
        
        # Pre-filter data columns more efficiently
        data_cols = [col for col in row1.index if col.startswith('data') and 
                    len(col) > 4 and col[4:].replace('0', '').isdigit()]
        
        if not data_cols:
            return changes
        
        # Vectorized comparison for all data columns at once
        for col in data_cols:
            if col in row2.index:
                val1, val2 = row1[col], row2[col]
                
                # Quick check if both are NaN (no change to track)
                if pd.isna(val1) and pd.isna(val2):
                    continue
                    
                # Compare as strings to catch any difference
                if str(val1) != str(val2):
                    changes[col] = {
                        'before': val1,
                        'after': val2,
                        'bus_before': row1['bus'],
                        'bus_after': row2['bus']
                    }
        
        return changes
    
    def analyze_message_patterns(self, df: pd.DataFrame, file_info: Dict):
        """
        Analyze message patterns and validate headers
        Optimized using vectorized operations for better performance
        """
        # Check if data01 column exists for header validation
        has_data01 = 'data01' in df.columns
        
        # Vectorized extraction of message types
        if 'decoded_description' in df.columns:
            # Process all decoded descriptions at once
            unique_msgs = df[['decoded_description', 'bus']].copy()
            if has_data01:
                unique_msgs['data01'] = df['data01']
            
            unique_msgs = unique_msgs.drop_duplicates()
            
            for _, row in unique_msgs.iterrows():
                msg_type, num_words = self.extract_message_type(row['decoded_description'])
                if msg_type:
                    key = (file_info['unit_id'], file_info['station'], file_info['save'])
                    
                    # Initialize sets if not present
                    if 'message_types' not in self.message_type_stats[key]:
                        self.message_type_stats[key]['message_types'] = set()
                    if 'buses' not in self.message_type_stats[key]:
                        self.message_type_stats[key]['buses'] = set()
                    if 'hyphenated_types' not in self.message_type_stats[key]:
                        self.message_type_stats[key]['hyphenated_types'] = set()
                    if 'header_mismatches' not in self.message_type_stats[key]:
                        self.message_type_stats[key]['header_mismatches'] = []
                    
                    self.message_type_stats[key]['message_types'].add(msg_type)
                    self.message_type_stats[key]['buses'].add(row['bus'])
                    
                    # Track hyphenated message types
                    if '-' in msg_type and not msg_type.endswith('R') and not msg_type.endswith('r'):
                        self.message_type_stats[key]['hyphenated_types'].add(msg_type)
                    
                    # Validate header if data01 exists
                    if has_data01:
                        is_valid, expected_headers = self.validate_header(msg_type, row.get('data01'))
                        if not is_valid:
                            self.message_type_stats[key]['header_mismatches'].append({
                                'msg_type': msg_type,
                                'actual_header': row.get('data01'),
                                'expected_headers': expected_headers
                            })
                            
                            # Add to global header issues tracking
                            self.header_issues.append({
                                'unit_id': file_info['unit_id'],
                                'station': file_info['station'],
                                'save': file_info['save'],
                                'msg_type': msg_type,
                                'actual_header': row.get('data01'),
                                'expected_headers': ', '.join(map(str, expected_headers)),
                                'bus': row['bus'],
                                'context': 'general_analysis'
                            })
    
    def process_csv(self, csv_path: Path) -> Dict:
        """
        Process a single CSV file with performance optimizations
        """
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            print(f"Warning: Could not parse filename {csv_path.name}")
            return None
        
        try:
            # Read CSV with optimizations
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Check required columns
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {csv_path.name}")
                return None
            
            # For large files, show progress
            if len(df) > 100000:
                print(f"  → Large file detected ({len(df):,} rows), using optimized processing...")
            
            # Detect bus flips (now vectorized)
            flips = self.detect_bus_flips(df, file_info)
            if flips:
                self.bus_flip_issues.extend(flips)
            
            # Analyze message patterns (now optimized)
            self.analyze_message_patterns(df, file_info)
            
            # Calculate bus statistics using value_counts (much faster)
            bus_counts = df['bus'].value_counts()
            
            return {
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'total_rows': len(df),
                'bus_flips': len(flips),
                'unique_buses': ', '.join(bus_counts.index.tolist()),
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0)
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
            # Add flip rate metric
            self.df_summary['flip_rate'] = (self.df_summary['bus_flips'] / self.df_summary['total_rows'] * 1000).round(2)
        
        # Flip issues
        if self.bus_flip_issues:
            self.df_flips = pd.DataFrame(self.bus_flip_issues)
            # Add header issue flag
            self.df_flips['has_header_issue'] = self.df_flips.apply(
                lambda x: x.get('header_issue_prev') is not None or x.get('header_issue_curr') is not None, 
                axis=1
            )
        
        # Data changes
        if self.data_changes:
            self.df_changes = pd.DataFrame(self.data_changes)
        
        # Header issues
        if self.header_issues:
            self.df_headers = pd.DataFrame(self.header_issues)
        else:
            self.df_headers = pd.DataFrame()
    
    def create_timeline_plot(self):
        """
        Create timeline scatter plot of bus flips
        """
        if self.df_flips is None or len(self.df_flips) == 0:
            return None
        
        fig = go.Figure()
        
        # Create traces for each unit_id
        for unit_id in self.df_flips['unit_id'].unique():
            unit_data = self.df_flips[self.df_flips['unit_id'] == unit_id]
            
            fig.add_trace(go.Scatter(
                x=unit_data['timestamp_prev'],
                y=unit_data['time_diff_ms'],
                mode='markers',
                name=unit_id,
                marker=dict(size=8),
                customdata=unit_data[['bus_transition', 'msg_type_prev', 'num_changes']],
                hovertemplate='<b>Unit: %{fullData.name}</b><br>' +
                             'Timestamp: %{x}<br>' +
                             'Time Diff: %{y:.2f} ms<br>' +
                             'Transition: %{customdata[0]}<br>' +
                             'Message: %{customdata[1]}<br>' +
                             'Changes: %{customdata[2]}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Timeline of Bus Flips",
            xaxis_title="Timestamp",
            yaxis_title="Time Difference (ms)",
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def create_heatmap(self):
        """
        Create heatmap of Unit ID vs Message Type
        """
        if self.df_flips is None or len(self.df_flips) == 0:
            return None
        
        # Create pivot table for heatmap
        heatmap_data = self.df_flips.pivot_table(
            index='unit_id',
            columns='msg_type_prev',
            values='time_diff_ms',
            aggfunc='count',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlOrRd',
            hovertemplate='Unit: %{y}<br>Message: %{x}<br>Flips: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Bus Flip Heatmap: Unit ID vs Message Type",
            xaxis_title="Message Type",
            yaxis_title="Unit ID",
            height=400
        )
        
        return fig
    
    def create_header_validation_plot(self):
        """
        Create visualization for header validation issues
        """
        if self.df_headers is None or len(self.df_headers) == 0:
            return None
        
        # Count header issues by message type
        header_counts = self.df_headers.groupby('msg_type').size().reset_index(name='count')
        header_counts = header_counts.nlargest(15, 'count')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=header_counts['msg_type'],
            y=header_counts['count'],
            marker_color='red',
            hovertemplate='<b>%{x}</b><br>Header Issues: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Header Validation Issues by Message Type",
            xaxis_title="Message Type",
            yaxis_title="Number of Issues",
            height=400
        )
        
        return fig
    
    def create_header_details_table(self):
        """
        Create detailed table of header mismatches
        """
        if self.df_headers is None or len(self.df_headers) == 0:
            return None
        
        # Get a sample of header issues for display
        sample_data = self.df_headers.head(100)[
            ['unit_id', 'station', 'msg_type', 'actual_header', 'expected_headers', 'bus']
        ].copy()
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Unit ID', 'Station', 'Message Type', 'Actual Header', 'Expected Headers', 'Bus'],
                fill_color='indianred',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[sample_data[col] for col in sample_data.columns],
                fill_color='mistyrose',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Sample of Header Validation Issues (First 100)",
            height=400
        )
        
        return fig
    
    def create_dashboard_with_filters(self):
        """
        Create enhanced dashboard with fully functional dropdown filters
        Using a cleaner approach with dynamic trace generation
        """
        if not self.df_summary is not None or len(self.df_summary) == 0:
            print("No data to visualize!")
            return None
        
        # Get unique unit IDs
        unit_ids = ['All'] + sorted(self.df_flips['unit_id'].unique().tolist()) if self.df_flips is not None else ['All']
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Bus Flips by Unit ID', 
                'Time Distribution of Flips',
                'Data Word Changes by Column', 
                'Message Type Distribution',
                'Bus Transition Patterns',
                'Files by Flip Rate (normalized)'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # Track number of traces per unit for visibility control
        traces_per_unit = 6
        
        # Generate traces for each unit (including 'All')
        for unit_idx, unit in enumerate(unit_ids):
            # Set initial visibility (only 'All' is visible at start)
            is_visible = (unit == 'All')
            
            # Filter dataframes based on unit selection
            if unit == 'All':
                df_flips_filtered = self.df_flips
                df_changes_filtered = self.df_changes
                df_summary_filtered = self.df_summary
            else:
                df_flips_filtered = self.df_flips[self.df_flips['unit_id'] == unit] if self.df_flips is not None else pd.DataFrame()
                df_changes_filtered = self.df_changes[self.df_changes['unit_id'] == unit] if self.df_changes is not None else pd.DataFrame()
                # For summary, filter files that contain this unit
                df_summary_filtered = self.df_summary[self.df_summary['unit_id'] == unit] if 'unit_id' in self.df_summary.columns else self.df_summary
            
            # 1. Bus Flips by Unit ID (only show for 'All')
            if unit == 'All' and self.df_flips is not None and len(self.df_flips) > 0:
                flip_counts = self.df_flips.groupby('unit_id').size().reset_index(name='count')
                fig.add_trace(
                    go.Bar(
                        x=flip_counts['unit_id'], 
                        y=flip_counts['count'],
                        name=f'Flips-{unit}',
                        marker_color='crimson',
                        visible=is_visible,
                        hovertemplate='<b>%{x}</b><br>Flips: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
            else:
                # Add placeholder for non-'All' units to maintain trace count
                if len(df_flips_filtered) > 0:
                    # Show single bar for selected unit
                    fig.add_trace(
                        go.Bar(
                            x=[unit],
                            y=[len(df_flips_filtered)],
                            name=f'Flips-{unit}',
                            marker_color='crimson',
                            visible=is_visible,
                            hovertemplate='<b>%{x}</b><br>Flips: %{y}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                else:
                    # Empty trace
                    fig.add_trace(go.Bar(x=[], y=[], visible=is_visible), row=1, col=1)
            
            # 2. Time Distribution
            if len(df_flips_filtered) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=df_flips_filtered['time_diff_ms'],
                        name=f'Time-{unit}',
                        nbinsx=20,
                        marker_color='steelblue',
                        visible=is_visible,
                        hovertemplate='Time: %{x} ms<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
            else:
                fig.add_trace(go.Histogram(x=[], visible=is_visible), row=1, col=2)
            
            # 3. Data Word Changes
            if len(df_changes_filtered) > 0:
                change_counts = df_changes_filtered['data_column'].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=change_counts.values,
                        y=change_counts.index,
                        orientation='h',
                        name=f'Changes-{unit}',
                        marker_color='orange',
                        visible=is_visible,
                        hovertemplate='%{y}<br>Changes: %{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
            else:
                fig.add_trace(go.Bar(x=[], y=[], orientation='h', visible=is_visible), row=2, col=1)
            
            # 4. Message Types
            if len(df_flips_filtered) > 0:
                msg_types = df_flips_filtered['msg_type_prev'].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=msg_types.index,
                        y=msg_types.values,
                        name=f'MsgTypes-{unit}',
                        marker_color='purple',
                        visible=is_visible,
                        hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=2
                )
            else:
                fig.add_trace(go.Bar(x=[], y=[], visible=is_visible), row=2, col=2)
            
            # 5. Bus Transition Patterns
            if len(df_flips_filtered) > 0:
                transition_counts = df_flips_filtered['bus_transition'].value_counts()
                fig.add_trace(
                    go.Bar(
                        x=transition_counts.index,
                        y=transition_counts.values,
                        name=f'Transitions-{unit}',
                        marker_color='teal',
                        visible=is_visible,
                        hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=3, col=1
                )
            else:
                fig.add_trace(go.Bar(x=[], y=[], visible=is_visible), row=3, col=1)
            
            # 6. Files by Flip Rate
            if len(df_summary_filtered) > 0 and 'flip_rate' in df_summary_filtered.columns:
                file_issues = df_summary_filtered.nlargest(min(10, len(df_summary_filtered)), 'flip_rate')
                fig.add_trace(
                    go.Bar(
                        x=file_issues['filename'],
                        y=file_issues['flip_rate'],
                        name=f'FlipRate-{unit}',
                        marker_color='darkgreen',
                        visible=is_visible,
                        customdata=file_issues[['bus_flips', 'total_rows']],
                        hovertemplate='<b>%{x}</b><br>' +
                                     'Flip Rate: %{y:.2f} per 1k rows<br>' +
                                     'Total Flips: %{customdata[0]}<br>' +
                                     'Total Rows: %{customdata[1]:,}<extra></extra>'
                    ),
                    row=3, col=2
                )
            else:
                fig.add_trace(go.Bar(x=[], y=[], visible=is_visible), row=3, col=2)
        
        # Create dropdown buttons with proper visibility arrays
        buttons = []
        for unit_idx, unit in enumerate(unit_ids):
            # Create visibility array for this unit
            visible = [False] * len(fig.data)
            
            # Set the 6 traces for this unit to visible
            start_idx = unit_idx * traces_per_unit
            end_idx = start_idx + traces_per_unit
            for i in range(start_idx, end_idx):
                if i < len(fig.data):
                    visible[i] = True
            
            buttons.append(
                dict(
                    label=unit,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"title": f"Bus Monitor Analysis Dashboard - {unit}"}
                    ]
                )
            )
        
        # Add dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            title_text="Bus Monitor Analysis Dashboard - All",
            height=1200,
            showlegend=False,
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Unit ID", row=1, col=1)
        fig.update_xaxes(title_text="Time Difference (ms)", row=1, col=2)
        fig.update_xaxes(title_text="Number of Changes", row=2, col=1)
        fig.update_xaxes(title_text="Message Type", row=2, col=2, tickangle=45)
        fig.update_xaxes(title_text="Bus Transition", row=3, col=1)
        fig.update_xaxes(title_text="Filename", row=3, col=2, tickangle=45)
        
        fig.update_yaxes(title_text="Flip Count", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Data Column", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_yaxes(title_text="Flip Rate (per 1k rows)", row=3, col=2)
        
        return fig
    
    def create_detailed_table(self):
        """
        Create detailed interactive table of all issues with header validation info
        """
        if self.df_flips is None or len(self.df_flips) == 0:
            return None
            
        # Prepare data for table
        table_data = self.df_flips[[
            'unit_id', 'station', 'save', 'timestamp_prev',
            'bus_transition', 'time_diff_ms', 'msg_type_prev', 
            'msg_type_curr', 'num_changes', 'has_header_issue'
        ]].copy()
        
        # Format timestamp for display
        table_data['timestamp_prev'] = table_data['timestamp_prev'].round(4)
        
        # Add header issue indicator
        table_data['has_header_issue'] = table_data['has_header_issue'].apply(
            lambda x: '⚠️ Yes' if x else '✓ No'
        )
        
        # Create row colors based on header issue status
        row_colors = []
        for val in table_data['has_header_issue']:
            if '⚠️' in str(val):
                row_colors.append('mistyrose')
            else:
                row_colors.append('lavender')
        
        # Create column colors by repeating row colors for each column
        # This ensures entire rows get the same color
        num_columns = len(table_data.columns)
        column_colors = [row_colors] * num_columns
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Unit ID', 'Station', 'Save', 'Timestamp', 
                       'Bus Trans.', 'Time (ms)', 'Msg Prev', 'Msg Curr', 
                       '# Changes', 'Header Issue'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[table_data[col] for col in table_data.columns],
                fill_color=column_colors,  # Use properly structured color list
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Detailed Bus Flip Issues with Header Validation",
            height=600
        )
        
        return fig
    
    def print_summary(self):
        """
        Print analysis summary to console including header validation results
        """
        print("\n" + "="*60)
        print("BUS MONITOR ANALYSIS SUMMARY")
        print("="*60)
        
        if self.df_summary is not None:
            print(f"\n📁 Files Processed: {len(self.df_summary)}")
            print(f"📊 Total Rows Analyzed: {self.df_summary['total_rows'].sum():,}")
        
        if self.df_flips is not None and len(self.df_flips) > 0:
            print(f"\n⚠️  Bus Flip Issues Found: {len(self.df_flips)}")
            print(f"📈 Average Time Between Flips: {self.df_flips['time_diff_ms'].mean():.2f} ms")
            print(f"⏱️  Min Time Between Flips: {self.df_flips['time_diff_ms'].min():.2f} ms")
            print(f"⏱️  Max Time Between Flips: {self.df_flips['time_diff_ms'].max():.2f} ms")
            
            # Header issue summary
            if 'has_header_issue' in self.df_flips.columns:
                header_issue_count = self.df_flips['has_header_issue'].sum()
                if header_issue_count > 0:
                    print(f"\n🔴 Header Validation Issues: {header_issue_count} flips with wrong headers")
            
            # Most problematic units
            print(f"\n🔥 Top 5 Most Problematic Units:")
            top_units = self.df_flips['unit_id'].value_counts().head(5)
            for unit, count in top_units.items():
                print(f"   • {unit}: {count} flips")
            
            # Most common message types with issues
            print(f"\n📨 Top 5 Message Types with Issues:")
            top_msgs = self.df_flips['msg_type_prev'].value_counts().head(5)
            for msg, count in top_msgs.items():
                print(f"   • {msg}: {count} flips")
        
        if self.df_headers is not None and len(self.df_headers) > 0:
            print(f"\n🔴 Header Validation Summary:")
            print(f"   • Total header mismatches: {len(self.df_headers)}")
            
            # Most common message types with header issues
            header_by_msg = self.df_headers['msg_type'].value_counts().head(5)
            print(f"\n   Message types with most header issues:")
            for msg_type, count in header_by_msg.items():
                # Get expected headers for this message type
                expected = self.message_header_lookup.get(msg_type, [])
                expected_str = ', '.join(map(str, expected)) if expected else 'N/A'
                print(f"   • {msg_type}: {count} issues (expected: {expected_str})")
        
        if self.df_changes is not None and len(self.df_changes) > 0:
            print(f"\n🔄 Total Data Word Changes: {len(self.df_changes)}")
            print(f"📊 Unique Data Columns Affected: {self.df_changes['data_column'].nunique()}")
            
            # Most frequently changed columns
            print(f"\n🎯 Most Frequently Changed Data Columns:")
            top_cols = self.df_changes['data_column'].value_counts().head(5)
            for col, count in top_cols.items():
                print(f"   • {col}: {count} changes")
        else:
            print("\n✅ No bus flip issues detected!")
        
        print("\n" + "="*60)
    
    def save_analysis_report(self):
        """
        Save enhanced HTML report with tabs, interactive features, and header validation
        """
        report_path = Path("bus_monitor_analysis_report.html")
        
        # Calculate KPIs
        total_files = len(self.df_summary) if self.df_summary is not None else 0
        total_rows = self.df_summary['total_rows'].sum() if self.df_summary is not None else 0
        total_flips = len(self.df_flips) if self.df_flips is not None else 0
        avg_flip_time = self.df_flips['time_diff_ms'].mean() if self.df_flips is not None and len(self.df_flips) > 0 else 0
        max_flip_rate = self.df_summary['flip_rate'].max() if self.df_summary is not None and 'flip_rate' in self.df_summary.columns else 0
        header_issues = len(self.df_headers) if self.df_headers is not None else 0
        
        # Determine severity including header issues
        if total_flips > 100 or header_issues > 50:
            severity_class = 'critical'
            severity_emoji = '🔴'
        elif total_flips > 20 or header_issues > 10:
            severity_class = 'warning'
            severity_emoji = '🟡'
        else:
            severity_class = 'good'
            severity_emoji = '🟢'
        
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
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                .subtitle {{
                    opacity: 0.9;
                    margin-top: 10px;
                }}
                .kpi-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    padding: 20px;
                    justify-content: center;
                }}
                .kpi-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    min-width: 180px;
                    text-align: center;
                    transition: transform 0.3s;
                }}
                .kpi-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }}
                .kpi-value {{
                    font-size: 36px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .kpi-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .critical {{ color: #e74c3c !important; }}
                .warning {{ color: #f39c12 !important; }}
                .good {{ color: #27ae60 !important; }}
                
                /* Tabs */
                .tabs {{
                    display: flex;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 0;
                    padding: 0 20px;
                }}
                .tab {{
                    padding: 15px 25px;
                    cursor: pointer;
                    border: none;
                    background: none;
                    font-size: 16px;
                    color: #7f8c8d;
                    transition: all 0.3s;
                    border-bottom: 3px solid transparent;
                }}
                .tab:hover {{
                    color: #2c3e50;
                }}
                .tab.active {{
                    color: #667eea;
                    border-bottom-color: #667eea;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    animation: fadeIn 0.3s;
                }}
                .tab-content.active {{
                    display: block;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px;
                }}
                .severity-badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .severity-badge.critical {{
                    background: #fee;
                    color: #e74c3c;
                }}
                .severity-badge.warning {{
                    background: #ffeaa7;
                    color: #f39c12;
                }}
                .severity-badge.good {{
                    background: #d1f2eb;
                    color: #27ae60;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚌 Bus Monitor Analysis Dashboard
                    <span class="severity-badge {severity_class}">{severity_emoji} {severity_class.upper()}</span>
                </h1>
                <div class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            
            <!-- KPI Cards -->
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Files Analyzed</div>
                    <div class="kpi-value">{total_files}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Rows</div>
                    <div class="kpi-value">{total_rows:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Bus Flips</div>
                    <div class="kpi-value {severity_class}">{total_flips}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Header Issues</div>
                    <div class="kpi-value {'critical' if header_issues > 50 else 'warning' if header_issues > 10 else 'good'}">{header_issues}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Flip Time</div>
                    <div class="kpi-value">{avg_flip_time:.1f} ms</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Max Flip Rate</div>
                    <div class="kpi-value">{max_flip_rate:.1f}</div>
                    <div style="font-size: 12px; color: #95a5a6;">per 1k rows</div>
                </div>
            </div>
            
            <!-- Tabs -->
            <div class="tabs">
                <button class="tab active" onclick="showTab(event, 'overview')">📊 Overview</button>
                <button class="tab" onclick="showTab(event, 'analysis')">🔍 Detailed Analysis</button>
                <button class="tab" onclick="showTab(event, 'headers')">🔴 Header Validation</button>
                <button class="tab" onclick="showTab(event, 'issues')">📋 Issue Log</button>
            </div>
            
            <!-- Tab Contents -->
            <div id="overview" class="tab-content active">
                <div class="chart-container">
        """
        
        # Add main dashboard
        dashboard_fig = self.create_dashboard_with_filters()
        if dashboard_fig:
            html_content += dashboard_fig.to_html(include_plotlyjs=False, div_id="dashboard")
        
        html_content += """
                </div>
            </div>
            
            <div id="analysis" class="tab-content">
        """
        
        # Add timeline plot
        timeline_fig = self.create_timeline_plot()
        if timeline_fig:
            html_content += '<div class="chart-container">'
            html_content += timeline_fig.to_html(include_plotlyjs=False, div_id="timeline")
            html_content += '</div>'
        
        # Add heatmap
        heatmap_fig = self.create_heatmap()
        if heatmap_fig:
            html_content += '<div class="chart-container">'
            html_content += heatmap_fig.to_html(include_plotlyjs=False, div_id="heatmap")
            html_content += '</div>'
        
        html_content += """
            </div>
            
            <div id="headers" class="tab-content">
        """
        
        # Add header validation visualizations
        header_plot = self.create_header_validation_plot()
        if header_plot:
            html_content += '<div class="chart-container">'
            html_content += header_plot.to_html(include_plotlyjs=False, div_id="header_plot")
            html_content += '</div>'
        
        header_table = self.create_header_details_table()
        if header_table:
            html_content += '<div class="chart-container">'
            html_content += header_table.to_html(include_plotlyjs=False, div_id="header_table")
            html_content += '</div>'
        else:
            html_content += '<div class="chart-container"><p>No header validation issues found or no lookup table provided.</p></div>'
        
        html_content += """
            </div>
            
            <div id="issues" class="tab-content">
                <div class="chart-container">
        """
        
        # Add detailed table
        table_fig = self.create_detailed_table()
        if table_fig:
            html_content += table_fig.to_html(include_plotlyjs=False, div_id="details")
        
        html_content += """
                </div>
            </div>
            
            <script>
                function showTab(evt, tabName) {
                    var i, tabcontent, tabs;
                    
                    // Hide all tab content
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].classList.remove("active");
                    }
                    
                    // Remove active class from all tabs
                    tabs = document.getElementsByClassName("tab");
                    for (i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove("active");
                    }
                    
                    // Show the selected tab content and mark the button as active
                    document.getElementById(tabName).classList.add("active");
                    evt.currentTarget.classList.add("active");
                }
            </script>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n📄 Enhanced interactive report saved to: {report_path.absolute()}")
        print("   → Open this file in your browser to view the interactive dashboard")


def main():
    """
    Main execution function - just run this!
    """
    print("🚀 Starting Enhanced Bus Monitor Analysis Dashboard")
    print("-" * 60)
    
    # Create analyzer
    analyzer = BusMonitorDashboard()
    
    # Run analysis
    results = analyzer.run_analysis()
    
    if not results:
        print("\n❌ No data was processed. Please check:")
        print(f"   1. The CSV folder path is correct: {analyzer.csv_folder}")
        print("   2. CSV files exist in that folder")
        print("   3. CSV files have the correct format")
        return
    
    # Print summary
    analyzer.print_summary()
    
    # Create and save enhanced dashboard
    print("\n📊 Generating enhanced interactive dashboard...")
    analyzer.save_analysis_report()
    
    print("\n✅ Analysis Complete!")
    print("   Open 'bus_monitor_analysis_report.html' in your browser to view the enhanced dashboard")


if __name__ == "__main__":
    main()
