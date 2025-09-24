#!/usr/bin/env python3
"""
Enhanced Bus Monitor Analysis - Version 15
Includes:
1. Data word analysis with common error patterns
2. DC1/DC2 state checking for valid bus flips
3. Tracking flips with no data changes separately
4. Bus flip percentage vs total messages
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class StreamlinedBusMonitorDashboard:
    def __init__(self):
        """Initialize the Streamlined Bus Monitor Dashboard"""
        # CONFIGURATION - Change these paths as needed
        self.csv_folder = Path("./csv_data")  # <-- UPDATE THIS PATH
        self.lookup_csv_path = Path("./message_lookup.csv")  # <-- PATH TO LOOKUP CSV
        self.requirements_folder = Path("./requirements")  # <-- PATH TO REQUIREMENTS EXCEL FILES
        self.output_folder = Path("./bus_monitor_output")  # <-- OUTPUT FOLDER
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Core data storage
        self.bus_flips = []
        self.bus_flips_no_changes = []  # Track flips with no data changes
        self.data_changes = []
        self.header_issues = []
        self.file_summary = []
        
        # Data word analysis storage
        self.data_word_issues = []  # Track all data word issues
        self.data_word_patterns = defaultdict(lambda: defaultdict(Counter))  # msg_type -> data_word -> error patterns
        
        # Summary storage
        self.unit_summary = []
        self.station_summary = []
        self.save_summary = []
        
        # Requirements analysis storage
        self.requirements_at_risk = []
        self.requirements_summary = []
        
        # Statistics tracking
        self.total_messages_processed = 0
        self.messages_with_dc_on = 0
        self.invalid_dc_messages = 0
        
        # DataFrames
        self.df_flips = None
        self.df_flips_no_changes = None
        self.df_data_changes = None
        self.df_header_issues = None
        self.df_file_summary = None
        self.df_unit_summary = None
        self.df_station_summary = None
        self.df_save_summary = None
        self.df_station_save_matrix = None
        self.df_requirements_at_risk = None
        self.df_requirements_summary = None
        self.df_data_word_analysis = None
        self.df_data_word_patterns = None
        
        # Load message lookup
        self.message_header_lookup = self.load_message_lookup()
    
    def load_message_lookup(self):
        """Load the message type to header lookup table"""
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
    
    def check_dc_states(self, row):
        """
        Check if DC1 or DC2 state is on (valid for bus flip detection)
        Returns True if at least one DC is on, False otherwise
        """
        dc1_on = False
        dc2_on = False
        
        if 'dc1_state' in row.index:
            dc1_val = str(row['dc1_state']).strip().upper()
            dc1_on = dc1_val in ['1', 'TRUE', 'ON', 'YES']
        
        if 'dc2_state' in row.index:
            dc2_val = str(row['dc2_state']).strip().upper()
            dc2_on = dc2_val in ['1', 'TRUE', 'ON', 'YES']
        
        return dc1_on or dc2_on
    
    def validate_header(self, msg_type: str, actual_header: str):
        """Validate if the actual header matches expected headers for message type"""
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
        """Parse filename to extract unit_id, station, save, and station_num"""
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
        """Extract message type from decoded description"""
        if pd.isna(decoded_desc):
            return None, 0
            
        pattern = r'\((\d+)-\[([^\]]+)\]-(\d+)\)'
        match = re.search(pattern, str(decoded_desc))
        
        if match:
            msg_type = match.group(2)
            end_num = int(match.group(3))
            return msg_type, end_num
        
        return None, 0
    
    def extract_message_type_from_column(self, col_name: str):
        """
        Extract message type from requirements column name
        Examples: [19r] -> 19R, [27r-2] -> 27R, [19r-3] -> 19R
        """
        # Match pattern like [XXr] or [XXr-Y] or [XXt] or [XXt-Y]
        match = re.search(r'\[(\d+)([rt])', str(col_name).lower())
        if match:
            return f"{match.group(1)}{match.group(2).upper()}"
        return None
    
    def load_requirements_files(self):
        """
        Load all requirements Excel files and extract relevant data
        """
        requirements_data = []
        
        if not self.requirements_folder.exists():
            print(f"Note: Requirements folder '{self.requirements_folder}' does not exist")
            return requirements_data
        
        excel_files = list(self.requirements_folder.glob("*_AllData.xlsx"))
        
        if not excel_files:
            print(f"No requirements Excel files found in {self.requirements_folder}")
            return requirements_data
        
        print(f"\nLoading {len(excel_files)} requirements files...")
        
        for excel_file in excel_files:
            try:
                # Extract requirement name from filename (e.g., ps-3000 from ps-3000_AllData.xlsx)
                requirement_name = excel_file.stem.replace('_AllData', '')
                
                # Read Excel file
                df = pd.read_excel(excel_file)
                
                # Check for required columns
                required_cols = ['unit_id', 'save', 'station']
                if not all(col in df.columns for col in required_cols):
                    print(f"  Warning: {excel_file.name} missing required columns")
                    continue
                
                # Find message type columns (columns with brackets)
                msg_type_cols = [col for col in df.columns if '[' in str(col) and ']' in str(col)]
                
                # Process each row
                for _, row in df.iterrows():
                    # Get base information
                    unit_id = str(row.get('unit_id', '')).strip()
                    save = str(row.get('save', '')).strip()
                    station = str(row.get('station', '')).strip()
                    ofp = str(row.get('ofp', '')).strip() if 'ofp' in df.columns else ''
                    
                    # Extract message types tested
                    msg_types_tested = []
                    for col in msg_type_cols:
                        msg_type = self.extract_message_type_from_column(col)
                        if msg_type and pd.notna(row[col]):
                            # Check if there's actual data in this column for this row
                            if str(row[col]).strip():
                                msg_types_tested.append(msg_type)
                    
                    if unit_id and save and station:
                        requirements_data.append({
                            'requirement_name': requirement_name,
                            'unit_id': unit_id,
                            'save': save,
                            'station': station,
                            'ofp': ofp,
                            'msg_types_tested': msg_types_tested,
                            'msg_types_str': ', '.join(sorted(set(msg_types_tested)))
                        })
                
                print(f"  Loaded {excel_file.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {excel_file.name}: {e}")
        
        print(f"  Total requirements records loaded: {len(requirements_data)}")
        return requirements_data
    
    def analyze_requirements_at_risk(self):
        """
        Cross-reference bus flips with requirements to identify affected requirements
        Only includes requirements where bus flips match their tested message types
        """
        if self.df_flips is None or self.df_flips.empty:
            print("No bus flips to analyze for requirements")
            return
        
        # Load requirements data
        requirements_data = self.load_requirements_files()
        
        if not requirements_data:
            print("No requirements data loaded")
            return
        
        # Create a lookup of bus flip issues by (unit_id, station, save, msg_type)
        flip_lookup = defaultdict(list)
        for _, flip in self.df_flips.iterrows():
            if flip.get('msg_type'):
                key = (str(flip['unit_id']), str(flip['station']), str(flip['save']), str(flip['msg_type']))
                flip_lookup[key].append({
                    'bus_transition': flip.get('bus_transition', ''),
                    'timestamp_busA': flip.get('timestamp_busA', 0),
                    'timestamp_busB': flip.get('timestamp_busB', 0)
                })
        
        # Check each requirement against bus flips
        # Use a set to track unique combinations and avoid duplicates
        seen_combinations = set()
        affected_requirements = []
        
        for req in requirements_data:
            # Get unique message types for this requirement
            unique_msg_types = list(set(req['msg_types_tested']))
            
            # Check each unique message type this requirement tests
            for msg_type in unique_msg_types:
                key = (req['unit_id'], req['station'], req['save'], msg_type)
                
                if key in flip_lookup:
                    # Create a unique identifier for this combination
                    combo_key = (req['requirement_name'], req['unit_id'], req['station'], 
                                 req['save'], msg_type)
                    
                    # Only add if we haven't seen this combination before
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        
                        # This requirement has bus flips for a message type it tests
                        flips_info = flip_lookup[key]
                        
                        affected_requirements.append({
                            'requirement_name': req['requirement_name'],
                            'unit_id': req['unit_id'],
                            'station': req['station'],
                            'save': req['save'],
                            'ofp': req['ofp'],
                            'msg_type_affected': msg_type,
                            'flip_count': len(flips_info),
                            'bus_transitions': ', '.join(sorted(set([f['bus_transition'] for f in flips_info])))
                        })
        
        # Create DataFrames
        if affected_requirements:
            self.df_requirements_at_risk = pd.DataFrame(affected_requirements)
            self.df_requirements_at_risk = self.df_requirements_at_risk.sort_values(
                ['requirement_name', 'flip_count'], 
                ascending=[True, False]
            )
            
            # Create summary by requirement
            req_summary = self.df_requirements_at_risk.groupby('requirement_name').agg({
                'msg_type_affected': lambda x: ', '.join(sorted(set(x))),
                'flip_count': 'sum',
                'unit_id': 'nunique',
                'station': 'nunique',
                'save': 'nunique'
            }).reset_index()
            
            req_summary.columns = ['requirement_name', 'affected_message_types', 'total_flips', 
                                  'unique_units', 'unique_stations', 'unique_saves']
            req_summary = req_summary.sort_values('total_flips', ascending=False)
            self.df_requirements_summary = req_summary
            
            print(f"\nRequirements Analysis Complete:")
            print(f"  Total affected requirement entries: {len(affected_requirements)}")
            print(f"  Unique requirements with issues: {len(self.df_requirements_summary)}")
    
    def analyze_data_word_patterns(self):
        """Analyze data word error patterns and create summaries"""
        if not self.data_word_issues:
            return
        
        # Create DataFrame from data word issues
        df_issues = pd.DataFrame(self.data_word_issues)
        
        # Group by message type and data word
        grouped = df_issues.groupby(['msg_type', 'data_word'])
        
        analysis_results = []
        for (msg_type, data_word), group in grouped:
            # Count unique error patterns (value_before -> value_after)
            error_patterns = group.apply(lambda x: f"{x['value_before']} -> {x['value_after']}", axis=1)
            pattern_counts = error_patterns.value_counts()
            
            # Get top 3 most common error patterns
            top_patterns = []
            for pattern, count in pattern_counts.head(3).items():
                top_patterns.append(f"{pattern} ({count}x)")
            
            analysis_results.append({
                'msg_type': msg_type,
                'data_word': data_word,
                'total_issues': len(group),
                'unique_patterns': len(pattern_counts),
                'top_error_patterns': ' | '.join(top_patterns),
                'affected_units': group['unit_id'].nunique(),
                'affected_stations': group['station'].nunique(),
                'affected_saves': group['save'].nunique()
            })
        
        self.df_data_word_analysis = pd.DataFrame(analysis_results)
        if not self.df_data_word_analysis.empty:
            self.df_data_word_analysis = self.df_data_word_analysis.sort_values('total_issues', ascending=False)
        
        # Create pattern frequency table
        pattern_results = []
        for (msg_type, data_word), group in grouped:
            for _, row in group.iterrows():
                pattern = f"{row['value_before']} -> {row['value_after']}"
                pattern_results.append({
                    'msg_type': msg_type,
                    'data_word': data_word,
                    'error_pattern': pattern,
                    'unit_id': row['unit_id'],
                    'station': row['station'],
                    'save': row['save']
                })
        
        if pattern_results:
            df_patterns = pd.DataFrame(pattern_results)
            pattern_freq = df_patterns.groupby(['msg_type', 'data_word', 'error_pattern']).size().reset_index(name='frequency')
            self.df_data_word_patterns = pattern_freq.sort_values(['msg_type', 'data_word', 'frequency'], ascending=[True, True, False])
    
    def detect_bus_flips(self, df: pd.DataFrame, file_info: dict):
        """Detect rapid bus flips with matching decoded_description"""
        flips = []
        flips_no_changes = []
        
        if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
            return flips
        
        # Check for DC state columns
        has_dc_columns = 'dc1_state' in df.columns or 'dc2_state' in df.columns
        
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
            
            # Check DC states if columns exist
            if has_dc_columns:
                prev_dc_valid = self.check_dc_states(prev_row)
                curr_dc_valid = self.check_dc_states(curr_row)
                
                # Skip if both DCs are off
                if not prev_dc_valid and not curr_dc_valid:
                    self.invalid_dc_messages += 2
                    continue
            
            msg_type, _ = self.extract_message_type(curr_row.get('decoded_description'))
            
            # Determine bus transition direction
            bus_transition = f"{prev_row['bus']} to {curr_row['bus']}"
            
            # Determine which timestamp belongs to which bus
            timestamp_busA = prev_row['timestamp'] if prev_row['bus'] == 'A' else curr_row['timestamp']
            timestamp_busB = prev_row['timestamp'] if prev_row['bus'] == 'B' else curr_row['timestamp']
            timestamp_diff = abs(curr_row['timestamp'] - prev_row['timestamp'])
            
            # Check for data changes
            data_changes = self.compare_data_words(prev_row, curr_row)
            has_data_changes = len(data_changes) > 0
            
            # Check header validation
            if 'data01' in df.columns and msg_type:
                is_valid_prev, expected = self.validate_header(msg_type, prev_row.get('data01'))
                is_valid_curr, _ = self.validate_header(msg_type, curr_row.get('data01'))
                
                if not is_valid_prev or not is_valid_curr:
                    self.header_issues.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'bus_transition': bus_transition,
                        'timestamp_busA': timestamp_busA,
                        'timestamp_busB': timestamp_busB,
                        'timestamp_diff': round(timestamp_diff, 6),
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
                'bus_transition': bus_transition,
                'timestamp_busA': timestamp_busA,
                'timestamp_busB': timestamp_busB,
                'timestamp_diff': round(timestamp_diff, 6),
                'msg_type': msg_type,
                'decoded_description': curr_row['decoded_description'],
                'has_data_changes': has_data_changes,
                'num_data_changes': len(data_changes)
            }
            
            # Separate tracking based on whether data changes occurred
            if has_data_changes:
                flips.append(flip_info)
                
                # Track data changes with enhanced info
                for col, change in data_changes.items():
                    value_busA = change['before'] if prev_row['bus'] == 'A' else change['after']
                    value_busB = change['before'] if prev_row['bus'] == 'B' else change['after']
                    
                    self.data_changes.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'bus_transition': bus_transition,
                        'timestamp_busA': timestamp_busA,
                        'timestamp_busB': timestamp_busB,
                        'timestamp_diff': round(timestamp_diff, 6),
                        'msg_type': msg_type,
                        'data_column': col,
                        'value_busA': value_busA,
                        'value_busB': value_busB
                    })
                    
                    # Track for data word analysis
                    self.data_word_issues.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'msg_type': msg_type,
                        'data_word': col,
                        'value_before': change['before'],
                        'value_after': change['after'],
                        'bus_before': change['bus_before'],
                        'bus_after': change['bus_after']
                    })
            else:
                # Track flips with no data changes separately
                flips_no_changes.append(flip_info)
        
        # Add flips with no changes to the separate list
        if flips_no_changes:
            self.bus_flips_no_changes.extend(flips_no_changes)
        
        return flips
    
    def compare_data_words(self, row1: pd.Series, row2: pd.Series):
        """Compare data word columns between two rows"""
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
    
    def process_csv(self, csv_path: Path):
        """Process a single CSV file"""
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            return None
        
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Track total messages
            self.total_messages_processed += len(df)
            
            # Track messages with DC on
            if 'dc1_state' in df.columns or 'dc2_state' in df.columns:
                dc_valid_count = df.apply(self.check_dc_states, axis=1).sum()
                self.messages_with_dc_on += dc_valid_count
            
            flips = self.detect_bus_flips(df, file_info)
            if flips:
                self.bus_flips.extend(flips)
            
            bus_counts = df['bus'].value_counts()
            
            # Count total flips (including those without data changes)
            flips_no_changes_count = len([f for f in self.bus_flips_no_changes 
                                         if f['unit_id'] == file_info['unit_id'] 
                                         and f['station'] == file_info['station'] 
                                         and f['save'] == file_info['save']])
            
            return {
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'total_rows': len(df),
                'bus_flips': len(flips),
                'bus_flips_no_changes': flips_no_changes_count,
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0),
                'unique_messages': df['decoded_description'].nunique() if 'decoded_description' in df.columns else 0
            }
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            return None
    
    def create_summaries(self):
        """Create summary dataframes at different levels"""
        if not self.df_file_summary.empty:
            # Unit ID Summary
            unit_agg = self.df_file_summary.groupby('unit_id').agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'station': 'nunique',
                'save': 'nunique'
            }).reset_index()
            
            unit_agg.columns = ['unit_id', 'total_rows', 'total_flips', 'flips_no_changes', 
                               'bus_a_total', 'bus_b_total', 'total_unique_messages', 
                               'file_count', 'station_count', 'save_count']
            
            self.df_unit_summary = unit_agg.sort_values('total_flips', ascending=False)
            
            # Station Summary (includes unit_id)
            station_agg = self.df_file_summary.groupby(['unit_id', 'station']).agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'save': 'nunique'
            }).reset_index()
            
            station_agg.columns = ['unit_id', 'station', 'total_rows', 'total_flips', 
                                  'flips_no_changes', 'bus_a_total', 'bus_b_total', 
                                  'total_unique_messages', 'file_count', 'save_count']
            
            self.df_station_summary = station_agg.sort_values('total_flips', ascending=False)
            
            # Save Summary (includes unit_id and station)
            save_agg = self.df_file_summary.groupby(['unit_id', 'station', 'save']).agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count'
            }).reset_index()
            
            save_agg.columns = ['unit_id', 'station', 'save', 'total_rows', 'total_flips', 
                               'flips_no_changes', 'bus_a_total', 'bus_b_total', 
                               'total_unique_messages', 'file_count']
            
            self.df_save_summary = save_agg.sort_values('total_flips', ascending=False)
    
    def create_station_save_matrix(self):
        """Create a matrix view of stations vs saves with flip counts per unit_id"""
        if self.df_flips is not None and not self.df_flips.empty:
            # Group by unit_id first, then create pivot for each
            matrices = {}
            for unit_id in self.df_flips['unit_id'].unique():
                unit_flips = self.df_flips[self.df_flips['unit_id'] == unit_id]
                matrix = unit_flips.groupby(['station', 'save']).size().reset_index(name='flip_count')
                pivot = matrix.pivot(index='station', columns='save', values='flip_count').fillna(0).astype(int)
                
                # Add totals
                pivot['TOTAL'] = pivot.sum(axis=1)
                pivot.loc['TOTAL'] = pivot.sum(axis=0)
                
                matrices[unit_id] = pivot
            
            # Combine all unit matrices (for the main matrix sheet)
            combined = self.df_flips.groupby(['station', 'save']).size().reset_index(name='flip_count')
            self.df_station_save_matrix = combined.pivot(index='station', columns='save', values='flip_count').fillna(0).astype(int)
            self.df_station_save_matrix['TOTAL'] = self.df_station_save_matrix.sum(axis=1)
            self.df_station_save_matrix.loc['TOTAL'] = self.df_station_save_matrix.sum(axis=0)
    
    def run_analysis(self):
        """Run the complete analysis on all CSV files"""
        if not self.csv_folder.exists():
            print(f"ERROR: CSV folder '{self.csv_folder}' does not exist!")
            return []
            
        csv_files = list(self.csv_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return []
            
        print(f"Found {len(csv_files)} CSV files to process")
        print("-" * 50)
        
        file_results = []
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Processing {csv_file.name}...")
            result = self.process_csv(csv_file)
            if result:
                file_results.append(result)
                self.file_summary.append(result)
        
        # Create DataFrames
        if self.file_summary:
            self.df_file_summary = pd.DataFrame(self.file_summary)
        
        if self.bus_flips:
            self.df_flips = pd.DataFrame(self.bus_flips)
        
        if self.bus_flips_no_changes:
            self.df_flips_no_changes = pd.DataFrame(self.bus_flips_no_changes)
        
        if self.data_changes:
            self.df_data_changes = pd.DataFrame(self.data_changes)
        
        if self.header_issues:
            self.df_header_issues = pd.DataFrame(self.header_issues)
        
        # Create summaries
        print("\nCreating summaries...")
        self.create_summaries()
        self.create_station_save_matrix()
        
        # Analyze data word patterns
        print("\nAnalyzing data word patterns...")
        self.analyze_data_word_patterns()
        
        # Analyze requirements at risk
        print("\nAnalyzing requirements at risk...")
        self.analyze_requirements_at_risk()
        
        return file_results
    
    def export_to_excel(self):
        """Export only essential data to Excel"""
        excel_path = self.output_folder / "bus_monitor_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1
            })
            
            # 1. Bus Flips Sheet (Main data - with data changes)
            if self.df_flips is not None and len(self.df_flips) > 0:
                self.df_flips.to_excel(writer, sheet_name='Bus_Flips', index=False)
                print(f"  Exported Bus Flips: {len(self.df_flips)} flips (with data changes)")
            
            # 2. Bus Flips No Changes Sheet
            if self.df_flips_no_changes is not None and len(self.df_flips_no_changes) > 0:
                self.df_flips_no_changes.to_excel(writer, sheet_name='Flips_No_Changes', index=False)
                print(f"  Exported Flips No Changes: {len(self.df_flips_no_changes)} flips")
            
            # 3. Data Word Analysis
            if self.df_data_word_analysis is not None and len(self.df_data_word_analysis) > 0:
                self.df_data_word_analysis.to_excel(writer, sheet_name='Data_Word_Analysis', index=False)
                print(f"  Exported Data Word Analysis: {len(self.df_data_word_analysis)} msg_type/data_word combinations")
            
            # 4. Data Word Patterns
            if self.df_data_word_patterns is not None and len(self.df_data_word_patterns) > 0:
                # Limit to top 1000 patterns if too many
                if len(self.df_data_word_patterns) > 1000:
                    self.df_data_word_patterns.head(1000).to_excel(writer, sheet_name='Data_Word_Patterns', index=False)
                    print(f"  Exported Data Word Patterns: 1000 patterns (truncated from {len(self.df_data_word_patterns)})")
                else:
                    self.df_data_word_patterns.to_excel(writer, sheet_name='Data_Word_Patterns', index=False)
                    print(f"  Exported Data Word Patterns: {len(self.df_data_word_patterns)} patterns")
            
            # 5. Requirements Affected
            if self.df_requirements_at_risk is not None and len(self.df_requirements_at_risk) > 0:
                self.df_requirements_at_risk.to_excel(writer, sheet_name='Requirements_Affected', index=False)
                print(f"  Exported Requirements Affected: {len(self.df_requirements_at_risk)} requirement instances")
            
            # 6. Requirements Summary
            if self.df_requirements_summary is not None and len(self.df_requirements_summary) > 0:
                self.df_requirements_summary.to_excel(writer, sheet_name='Requirements_Summary', index=False)
                print(f"  Exported Requirements Summary: {len(self.df_requirements_summary)} unique requirements")
            
            # 7. Header Issues
            if self.df_header_issues is not None and len(self.df_header_issues) > 0:
                self.df_header_issues.to_excel(writer, sheet_name='Header_Issues', index=False)
                print(f"  Exported Header Issues: {len(self.df_header_issues)} issues")
            
            # 8. Data Changes
            if self.df_data_changes is not None and len(self.df_data_changes) > 0:
                # Limit to first 10000 rows if too many
                if len(self.df_data_changes) > 10000:
                    self.df_data_changes.iloc[:10000].to_excel(writer, sheet_name='Data_Changes', index=False)
                    print(f"  Exported Data Changes: 10000 rows (truncated from {len(self.df_data_changes)})")
                else:
                    self.df_data_changes.to_excel(writer, sheet_name='Data_Changes', index=False)
                    print(f"  Exported Data Changes: {len(self.df_data_changes)} rows")
            
            # 9. File Summary
            if self.df_file_summary is not None and len(self.df_file_summary) > 0:
                self.df_file_summary.to_excel(writer, sheet_name='File_Summary', index=False)
                print(f"  Exported File Summary: {len(self.df_file_summary)} files")
            
            # 10. Station-Save Matrix
            if self.df_station_save_matrix is not None and len(self.df_station_save_matrix) > 0:
                self.df_station_save_matrix.to_excel(writer, sheet_name='Station_Save_Matrix')
                print(f"  Exported Station-Save Matrix")
            
            # 11. Unit ID Summary
            if self.df_unit_summary is not None and len(self.df_unit_summary) > 0:
                self.df_unit_summary.to_excel(writer, sheet_name='Unit_Summary', index=False)
                print(f"  Exported Unit Summary: {len(self.df_unit_summary)} units")
            
            # 12. Station Summary
            if self.df_station_summary is not None and len(self.df_station_summary) > 0:
                self.df_station_summary.to_excel(writer, sheet_name='Station_Summary', index=False)
                print(f"  Exported Station Summary: {len(self.df_station_summary)} stations")
            
            # 13. Save Summary
            if self.df_save_summary is not None and len(self.df_save_summary) > 0:
                self.df_save_summary.to_excel(writer, sheet_name='Save_Summary', index=False)
                print(f"  Exported Save Summary: {len(self.df_save_summary)} saves")
        
        print(f"\nExcel file saved to: {excel_path.absolute()}")
        return excel_path
    
    def create_interactive_dashboard(self):
        """Create an interactive HTML dashboard with filters"""
        dashboard_path = self.output_folder / "dashboard.html"
        
        # Prepare data for JavaScript
        flips_data = []
        if self.df_flips is not None and not self.df_flips.empty:
            flips_data = self.df_flips.to_dict('records')
        
        # Get unique values for filters
        unit_ids = sorted(self.df_flips['unit_id'].unique().tolist()) if self.df_flips is not None else []
        stations = sorted(self.df_flips['station'].unique().tolist()) if self.df_flips is not None else []
        saves = sorted(self.df_flips['save'].unique().tolist()) if self.df_flips is not None else []
        msg_types = sorted(self.df_flips['msg_type'].dropna().unique().tolist()) if self.df_flips is not None else []
        
        # Calculate summary stats
        total_flips = len(self.df_flips) if self.df_flips is not None else 0
        total_flips_no_changes = len(self.df_flips_no_changes) if self.df_flips_no_changes is not None else 0
        total_units = len(unit_ids)
        total_stations = len(stations)
        total_saves = len(saves)
        
        # Calculate flip percentage
        flip_percentage = 0
        if self.total_messages_processed > 0:
            flip_percentage = (total_flips / self.total_messages_processed) * 100
        
        # Calculate R vs T message counts
        r_messages = 0
        t_messages = 0
        other_messages = 0
        if self.df_flips is not None and not self.df_flips.empty:
            for msg_type in self.df_flips['msg_type'].dropna():
                msg_str = str(msg_type).strip()
                # Extract pattern like 19R, 27T, etc.
                if re.match(r'^\d+R', msg_str):
                    r_messages += 1
                elif re.match(r'^\d+T', msg_str):
                    t_messages += 1
                else:
                    other_messages += 1
        
        # Get data word analysis for dashboard
        data_word_data = []
        if self.df_data_word_analysis is not None and not self.df_data_word_analysis.empty:
            data_word_data = self.df_data_word_analysis.head(20).to_dict('records')
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bus Monitor Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 15px;
        }}
        .filters {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .filter-group {{
            display: flex;
            flex-direction: column;
        }}
        .filter-group label {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }}
        .filter-group select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            font-size: 14px;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.warning {{
            background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
        }}
        .stat-card.info {{
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .stat-label {{
            margin-top: 5px;
            opacity: 0.9;
            font-size: 14px;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        #filteredCount {{
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            margin: 20px 0;
            font-weight: bold;
        }}
        .reset-btn {{
            background: #ff9800;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }}
        .reset-btn:hover {{
            background: #e68900;
        }}
        .data-word-table {{
            margin-top: 20px;
            width: 100%;
            border-collapse: collapse;
        }}
        .data-word-table th, .data-word-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .data-word-table th {{
            background: #f5f5f5;
            font-weight: bold;
        }}
        .data-word-table tr:hover {{
            background: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bus Monitor Interactive Dashboard</h1>
        
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{total_flips:,}</div>
                <div class="stat-label">Bus Flips (with changes)</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{total_flips_no_changes:,}</div>
                <div class="stat-label">Flips (no changes)</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{flip_percentage:.3f}%</div>
                <div class="stat-label">Flip Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.total_messages_processed:,}</div>
                <div class="stat-label">Total Messages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_units}</div>
                <div class="stat-label">Unit IDs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_stations}</div>
                <div class="stat-label">Stations</div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label for="unitFilter">Unit ID:</label>
                <select id="unitFilter" onchange="updateFilters()">
                    <option value="">All Units</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="stationFilter">Station:</label>
                <select id="stationFilter" onchange="updateFilters()">
                    <option value="">All Stations</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="saveFilter">Save:</label>
                <select id="saveFilter" onchange="updateFilters()">
                    <option value="">All Saves</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="msgTypeFilter">Message Type:</label>
                <select id="msgTypeFilter" onchange="updateFilters()">
                    <option value="">All Message Types</option>
                </select>
            </div>
        </div>
        
        <div>
            <span id="filteredCount">Showing all {total_flips} flips</span>
            <button class="reset-btn" onclick="resetFilters()">Reset All Filters</button>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Unit ID</div>
            <div id="unitChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Station</div>
            <div id="stationChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Save</div>
            <div id="saveChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">R vs T Message Distribution</div>
            <div id="rtChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Message Type</div>
            <div id="msgTypeChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Top Data Word Issues</div>
            <div id="dataWordChart"></div>
            <table class="data-word-table" id="dataWordTable">
                <thead>
                    <tr>
                        <th>Message Type</th>
                        <th>Data Word</th>
                        <th>Issues</th>
                        <th>Top Error Patterns</th>
                    </tr>
                </thead>
                <tbody id="dataWordTableBody">
                </tbody>
            </table>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Hierarchical View: Unit → Station → Save</div>
            <div id="hierarchicalChart"></div>
        </div>
    </div>
    
    <script>
        // Data from Python
        const allData = {flips_data};
        const dataWordData = {data_word_data};
        let filteredData = [...allData];
        
        // Unique values for filters
        const uniqueUnits = {unit_ids};
        const uniqueStations = {stations};
        const uniqueSaves = {saves};
        const uniqueMsgTypes = {msg_types};
        
        // Initialize filters
        function initializeFilters() {{
            const unitFilter = document.getElementById('unitFilter');
            uniqueUnits.forEach(unit => {{
                const option = document.createElement('option');
                option.value = unit;
                option.textContent = unit;
                unitFilter.appendChild(option);
            }});
            
            const stationFilter = document.getElementById('stationFilter');
            uniqueStations.forEach(station => {{
                const option = document.createElement('option');
                option.value = station;
                option.textContent = station;
                stationFilter.appendChild(option);
            }});
            
            const saveFilter = document.getElementById('saveFilter');
            uniqueSaves.forEach(save => {{
                const option = document.createElement('option');
                option.value = save;
                option.textContent = save;
                saveFilter.appendChild(option);
            }});
            
            const msgTypeFilter = document.getElementById('msgTypeFilter');
            uniqueMsgTypes.forEach(msgType => {{
                const option = document.createElement('option');
                option.value = msgType;
                option.textContent = msgType;
                msgTypeFilter.appendChild(option);
            }});
        }}
        
        function updateFilters() {{
            const unitFilter = document.getElementById('unitFilter').value;
            const stationFilter = document.getElementById('stationFilter').value;
            const saveFilter = document.getElementById('saveFilter').value;
            const msgTypeFilter = document.getElementById('msgTypeFilter').value;
            
            // Filter data
            filteredData = allData.filter(row => {{
                return (!unitFilter || row.unit_id === unitFilter) &&
                       (!stationFilter || row.station === stationFilter) &&
                       (!saveFilter || row.save === saveFilter) &&
                       (!msgTypeFilter || row.msg_type === msgTypeFilter);
            }});
            
            // Update filtered count
            document.getElementById('filteredCount').textContent = 
                `Showing ${{filteredData.length}} of ${{allData.length}} flips`;
            
            // Update available options based on current selection
            updateAvailableOptions();
            
            // Redraw charts
            drawCharts();
        }}
        
        function updateAvailableOptions() {{
            // Get current selections
            const unitFilter = document.getElementById('unitFilter').value;
            const stationFilter = document.getElementById('stationFilter').value;
            const saveFilter = document.getElementById('saveFilter').value;
            
            // Update station options based on selected unit
            if (unitFilter) {{
                const availableStations = [...new Set(filteredData.map(d => d.station))];
                updateSelectOptions('stationFilter', availableStations, stationFilter);
            }}
            
            // Update save options based on selected unit and station
            if (unitFilter || stationFilter) {{
                const availableSaves = [...new Set(filteredData.map(d => d.save))];
                updateSelectOptions('saveFilter', availableSaves, saveFilter);
            }}
        }}
        
        function updateSelectOptions(selectId, options, currentValue) {{
            const select = document.getElementById(selectId);
            const previousValue = select.value;
            
            // Remove all options except the first (All)
            while (select.options.length > 1) {{
                select.remove(1);
            }}
            
            // Add new options
            options.sort().forEach(option => {{
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                select.appendChild(optionElement);
            }});
            
            // Restore previous selection if it's still valid
            if (options.includes(previousValue)) {{
                select.value = previousValue;
            }}
        }}
        
        function resetFilters() {{
            document.getElementById('unitFilter').value = '';
            document.getElementById('stationFilter').value = '';
            document.getElementById('saveFilter').value = '';
            document.getElementById('msgTypeFilter').value = '';
            updateFilters();
        }}
        
        function drawCharts() {{
            // Unit ID Chart
            const unitCounts = {{}};
            filteredData.forEach(d => {{
                unitCounts[d.unit_id] = (unitCounts[d.unit_id] || 0) + 1;
            }});
            
            Plotly.newPlot('unitChart', [{{
                x: Object.keys(unitCounts),
                y: Object.values(unitCounts),
                type: 'bar',
                marker: {{ color: '#667eea' }}
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                xaxis: {{ title: 'Unit ID' }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // Station Chart
            const stationCounts = {{}};
            filteredData.forEach(d => {{
                stationCounts[d.station] = (stationCounts[d.station] || 0) + 1;
            }});
            
            Plotly.newPlot('stationChart', [{{
                x: Object.keys(stationCounts),
                y: Object.values(stationCounts),
                type: 'bar',
                marker: {{ color: '#764ba2' }}
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                xaxis: {{ title: 'Station' }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // Save Chart
            const saveCounts = {{}};
            filteredData.forEach(d => {{
                saveCounts[d.save] = (saveCounts[d.save] || 0) + 1;
            }});
            
            Plotly.newPlot('saveChart', [{{
                x: Object.keys(saveCounts),
                y: Object.values(saveCounts),
                type: 'bar',
                marker: {{ color: '#4CAF50' }}
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                xaxis: {{ title: 'Save' }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // R vs T Message Chart
            const rCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+R/.test(msgStr);  // Match patterns like 19R, 27R, etc.
            }}).length;
            
            const tCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+T/.test(msgStr);  // Match patterns like 19T, 27T, etc.
            }}).length;
            
            // Bar chart for R vs T distribution
            Plotly.newPlot('rtChart', [{{
                x: ['R Messages', 'T Messages'],
                y: [rCount, tCount],
                type: 'bar',
                marker: {{
                    color: ['#4CAF50', '#2196F3']
                }},
                text: [rCount, tCount],
                textposition: 'auto',
                hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                height: 350,
                yaxis: {{ title: 'Count' }},
                showlegend: false
            }});
            
            // Message Type Chart
            const msgTypeCounts = {{}};
            filteredData.forEach(d => {{
                if (d.msg_type) {{
                    msgTypeCounts[d.msg_type] = (msgTypeCounts[d.msg_type] || 0) + 1;
                }}
            }});
            
            // Sort by count and take top 20
            const sortedMsgTypes = Object.entries(msgTypeCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 20);
            
            Plotly.newPlot('msgTypeChart', [{{
                x: sortedMsgTypes.map(x => x[0]),
                y: sortedMsgTypes.map(x => x[1]),
                type: 'bar',
                marker: {{ color: '#ff9800' }}
            }}], {{
                margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                xaxis: {{ title: 'Message Type', tickangle: -45 }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // Data Word Chart
            if (dataWordData.length > 0) {{
                Plotly.newPlot('dataWordChart', [{{
                    x: dataWordData.map(d => `${{d.msg_type}}-${{d.data_word}}`),
                    y: dataWordData.map(d => d.total_issues),
                    type: 'bar',
                    marker: {{ color: '#e91e63' }}
                }}], {{
                    margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Type - Data Word', tickangle: -45 }},
                    yaxis: {{ title: 'Number of Issues' }}
                }});
                
                // Populate data word table
                const tableBody = document.getElementById('dataWordTableBody');
                tableBody.innerHTML = '';
                dataWordData.forEach(d => {{
                    const row = tableBody.insertRow();
                    row.insertCell(0).textContent = d.msg_type || 'N/A';
                    row.insertCell(1).textContent = d.data_word;
                    row.insertCell(2).textContent = d.total_issues;
                    row.insertCell(3).textContent = d.top_error_patterns || 'N/A';
                }});
            }}
            
            // Hierarchical Chart (Sunburst)
            const hierarchicalData = [];
            const labels = [];
            const parents = [];
            const values = [];
            const colors = [];
            
            // Add root
            labels.push('All');
            parents.push('');
            values.push(filteredData.length);
            colors.push('#667eea');
            
            // Group by unit
            const unitGroups = {{}};
            filteredData.forEach(d => {{
                if (!unitGroups[d.unit_id]) {{
                    unitGroups[d.unit_id] = {{}};
                }}
                if (!unitGroups[d.unit_id][d.station]) {{
                    unitGroups[d.unit_id][d.station] = {{}};
                }}
                if (!unitGroups[d.unit_id][d.station][d.save]) {{
                    unitGroups[d.unit_id][d.station][d.save] = 0;
                }}
                unitGroups[d.unit_id][d.station][d.save]++;
            }});
            
            // Build hierarchy
            Object.keys(unitGroups).forEach(unit => {{
                labels.push(unit);
                parents.push('All');
                values.push(Object.values(unitGroups[unit]).reduce((sum, stations) => 
                    sum + Object.values(stations).reduce((s, v) => s + v, 0), 0));
                colors.push('#764ba2');
                
                Object.keys(unitGroups[unit]).forEach(station => {{
                    const stationLabel = `${{unit}} - ${{station}}`;
                    labels.push(stationLabel);
                    parents.push(unit);
                    values.push(Object.values(unitGroups[unit][station]).reduce((s, v) => s + v, 0));
                    colors.push('#4CAF50');
                    
                    Object.keys(unitGroups[unit][station]).forEach(save => {{
                        labels.push(`${{stationLabel}} - ${{save}}`);
                        parents.push(stationLabel);
                        values.push(unitGroups[unit][station][save]);
                        colors.push('#ff9800');
                    }});
                }});
            }});
            
            Plotly.newPlot('hierarchicalChart', [{{
                type: 'sunburst',
                labels: labels,
                parents: parents,
                values: values,
                marker: {{ colors: colors }}
            }}], {{
                margin: {{ t: 10, b: 10, l: 10, r: 10 }},
                height: 600
            }});
        }}
        
        // Initialize on page load
        initializeFilters();
        drawCharts();
    </script>
</body>
</html>
"""
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nInteractive dashboard saved to: {dashboard_path.absolute()}")
        return dashboard_path
