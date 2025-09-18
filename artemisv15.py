#!/usr/bin/env python3
"""
Multi-Test Bus Monitor Analysis - Processes multiple test folders
Creates individual and combined analysis reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class MultiTestBusMonitorDashboard:
    def __init__(self):
        """Initialize the Multi-Test Bus Monitor Dashboard"""
        # CONFIGURATION - Change these paths as needed
        self.parent_folder = Path("./test_data")  # <-- Parent folder containing test subfolders
        self.lookup_csv_path = Path("./message_lookup.csv")  # <-- PATH TO LOOKUP CSV
        self.output_folder = Path("./bus_monitor_output")  # <-- OUTPUT FOLDER
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Storage for all tests combined
        self.all_tests_data = {}
        self.combined_flips = []
        self.combined_file_summary = []
        self.combined_header_issues = []
        self.combined_data_changes = []
        
        # DataFrames for combined analysis
        self.df_combined_flips = None
        self.df_combined_summary = None
        self.df_test_comparison = None
        
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
        
        return lookup
    
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
    
    def detect_bus_flips(self, df: pd.DataFrame, file_info: dict, test_name: str):
        """Detect rapid bus flips with matching decoded_description"""
        flips = []
        header_issues = []
        data_changes = []
        
        if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
            return flips, header_issues, data_changes
            
        df = df.copy()
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            return flips, header_issues, data_changes
        
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
            
            # Determine bus transition direction
            bus_transition = f"{prev_row['bus']} to {curr_row['bus']}"
            
            # Determine which timestamp belongs to which bus
            timestamp_busA = prev_row['timestamp'] if prev_row['bus'] == 'A' else curr_row['timestamp']
            timestamp_busB = prev_row['timestamp'] if prev_row['bus'] == 'B' else curr_row['timestamp']
            timestamp_diff = abs(curr_row['timestamp'] - prev_row['timestamp'])
            
            # Check header validation
            if 'data01' in df.columns and msg_type:
                is_valid_prev, expected = self.validate_header(msg_type, prev_row.get('data01'))
                is_valid_curr, _ = self.validate_header(msg_type, curr_row.get('data01'))
                
                if not is_valid_prev or not is_valid_curr:
                    header_issues.append({
                        'test_name': test_name,
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
            
            # Create flip record with test name
            flip_info = {
                'test_name': test_name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'bus_transition': bus_transition,
                'timestamp_busA': timestamp_busA,
                'timestamp_busB': timestamp_busB,
                'timestamp_diff': round(timestamp_diff, 6),
                'msg_type': msg_type,
                'decoded_description': curr_row['decoded_description']
            }
            
            flips.append(flip_info)
            
            # Track data changes with test name
            data_diffs = self.compare_data_words(prev_row, curr_row)
            for col, change in data_diffs.items():
                data_changes.append({
                    'test_name': test_name,
                    'unit_id': file_info['unit_id'],
                    'station': file_info['station'],
                    'save': file_info['save'],
                    'bus_transition': bus_transition,
                    'timestamp_busA': timestamp_busA,
                    'timestamp_busB': timestamp_busB,
                    'timestamp_diff': round(timestamp_diff, 6),
                    'msg_type': msg_type,
                    'data_column': col,
                    'value_busA': change['before'] if prev_row['bus'] == 'A' else change['after'],
                    'value_busB': change['before'] if prev_row['bus'] == 'B' else change['after']
                })
        
        return flips, header_issues, data_changes
    
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
    
    def process_csv(self, csv_path: Path, test_name: str):
        """Process a single CSV file"""
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            return None, [], [], []
        
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None, [], [], []
            
            flips, header_issues, data_changes = self.detect_bus_flips(df, file_info, test_name)
            
            bus_counts = df['bus'].value_counts()
            
            return {
                'test_name': test_name,
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'total_rows': len(df),
                'bus_flips': len(flips),
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0),
                'unique_messages': df['decoded_description'].nunique() if 'decoded_description' in df.columns else 0
            }, flips, header_issues, data_changes
            
        except Exception as e:
            print(f"    Error processing {csv_path.name}: {e}")
            return None, [], [], []
    
    def process_test_folder(self, test_folder: Path):
        """Process a single test folder containing CSV files"""
        test_name = test_folder.name
        print(f"\nProcessing test: {test_name}")
        print("-" * 40)
        
        csv_files = list(test_folder.glob("*.csv"))
        if not csv_files:
            print(f"  No CSV files found in {test_folder}")
            return None
        
        print(f"  Found {len(csv_files)} CSV files")
        
        test_data = {
            'test_name': test_name,
            'file_summary': [],
            'bus_flips': [],
            'header_issues': [],
            'data_changes': []
        }
        
        for csv_file in csv_files:
            file_result, flips, header_issues, data_changes = self.process_csv(csv_file, test_name)
            
            if file_result:
                test_data['file_summary'].append(file_result)
                test_data['bus_flips'].extend(flips)
                test_data['header_issues'].extend(header_issues)
                test_data['data_changes'].extend(data_changes)
        
        # Create summary statistics for this test
        test_data['summary'] = {
            'test_name': test_name,
            'total_files': len(test_data['file_summary']),
            'total_flips': len(test_data['bus_flips']),
            'total_header_issues': len(test_data['header_issues']),
            'total_data_changes': len(test_data['data_changes']),
            'unique_units': len(set(f['unit_id'] for f in test_data['file_summary'])),
            'unique_stations': len(set(f['station'] for f in test_data['file_summary'])),
            'unique_saves': len(set(f['save'] for f in test_data['file_summary']))
        }
        
        print(f"  Completed: {test_data['summary']['total_flips']} flips detected")
        
        return test_data
    
    def run_analysis(self):
        """Run analysis on all test folders"""
        if not self.parent_folder.exists():
            print(f"ERROR: Parent folder '{self.parent_folder}' does not exist!")
            return
        
        # Find all subdirectories in parent folder
        test_folders = [f for f in self.parent_folder.iterdir() if f.is_dir()]
        
        if not test_folders:
            print(f"No test folders found in {self.parent_folder}")
            return
        
        print(f"Found {len(test_folders)} test folders to process")
        print("=" * 60)
        
        # Process each test folder
        for test_folder in sorted(test_folders):
            test_data = self.process_test_folder(test_folder)
            
            if test_data and test_data['bus_flips']:
                self.all_tests_data[test_data['test_name']] = test_data
                
                # Add to combined data
                self.combined_flips.extend(test_data['bus_flips'])
                self.combined_file_summary.extend(test_data['file_summary'])
                self.combined_header_issues.extend(test_data['header_issues'])
                self.combined_data_changes.extend(test_data['data_changes'])
        
        # Create combined DataFrames
        if self.combined_flips:
            self.df_combined_flips = pd.DataFrame(self.combined_flips)
        
        if self.combined_file_summary:
            self.df_combined_summary = pd.DataFrame(self.combined_file_summary)
        
        # Create test comparison summary
        self.create_test_comparison_summary()
        
        print("\n" + "=" * 60)
        print(f"Analysis complete! Processed {len(self.all_tests_data)} tests")
    
    def create_test_comparison_summary(self):
        """Create a summary comparing all tests"""
        if not self.all_tests_data:
            return
        
        comparison_data = []
        for test_name, test_data in self.all_tests_data.items():
            summary = test_data['summary'].copy()
            
            # Add R vs T message counts
            r_count = 0
            t_count = 0
            other_count = 0
            
            for flip in test_data['bus_flips']:
                msg_type = flip.get('msg_type', '')
                if re.match(r'^\d+R', str(msg_type)):
                    r_count += 1
                elif re.match(r'^\d+T', str(msg_type)):
                    t_count += 1
                else:
                    other_count += 1
            
            summary['r_messages'] = r_count
            summary['t_messages'] = t_count
            summary['other_messages'] = other_count
            summary['r_percentage'] = round(r_count / len(test_data['bus_flips']) * 100, 1) if test_data['bus_flips'] else 0
            summary['t_percentage'] = round(t_count / len(test_data['bus_flips']) * 100, 1) if test_data['bus_flips'] else 0
            
            comparison_data.append(summary)
        
        self.df_test_comparison = pd.DataFrame(comparison_data)
        self.df_test_comparison = self.df_test_comparison.sort_values('total_flips', ascending=False)
    
    def export_individual_excels(self):
        """Export individual Excel files for each test"""
        for test_name, test_data in self.all_tests_data.items():
            excel_path = self.output_folder / f"{test_name}_analysis.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Bus Flips
                if test_data['bus_flips']:
                    df = pd.DataFrame(test_data['bus_flips'])
                    df.to_excel(writer, sheet_name='Bus_Flips', index=False)
                
                # File Summary
                if test_data['file_summary']:
                    df = pd.DataFrame(test_data['file_summary'])
                    df.to_excel(writer, sheet_name='File_Summary', index=False)
                
                # Header Issues
                if test_data['header_issues']:
                    df = pd.DataFrame(test_data['header_issues'])
                    df.to_excel(writer, sheet_name='Header_Issues', index=False)
                
                # Data Changes (limited to 10000 rows)
                if test_data['data_changes']:
                    df = pd.DataFrame(test_data['data_changes'])
                    if len(df) > 10000:
                        df = df.iloc[:10000]
                    df.to_excel(writer, sheet_name='Data_Changes', index=False)
            
            print(f"  Exported: {excel_path.name}")
    
    def export_super_excel(self):
        """Export combined Excel with all tests data"""
        excel_path = self.output_folder / "super_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Test Comparison Summary (first sheet)
            if self.df_test_comparison is not None:
                self.df_test_comparison.to_excel(writer, sheet_name='Test_Comparison', index=False)
                print(f"  Exported Test Comparison: {len(self.df_test_comparison)} tests")
            
            # Combined Bus Flips
            if self.df_combined_flips is not None and len(self.df_combined_flips) > 0:
                self.df_combined_flips.to_excel(writer, sheet_name='All_Bus_Flips', index=False)
                print(f"  Exported All Bus Flips: {len(self.df_combined_flips)} flips")
            
            # Combined File Summary
            if self.df_combined_summary is not None and len(self.df_combined_summary) > 0:
                self.df_combined_summary.to_excel(writer, sheet_name='All_File_Summary', index=False)
                print(f"  Exported All File Summary: {len(self.df_combined_summary)} files")
            
            # Combined Header Issues
            if self.combined_header_issues:
                df = pd.DataFrame(self.combined_header_issues)
                df.to_excel(writer, sheet_name='All_Header_Issues', index=False)
                print(f"  Exported All Header Issues: {len(df)} issues")
            
            # Combined Data Changes (limited)
            if self.combined_data_changes:
                df = pd.DataFrame(self.combined_data_changes)
                if len(df) > 20000:
                    df = df.iloc[:20000]
                    print(f"  Exported All Data Changes: 20000 rows (truncated from {len(self.combined_data_changes)})")
                else:
                    print(f"  Exported All Data Changes: {len(df)} rows")
                df.to_excel(writer, sheet_name='All_Data_Changes', index=False)
            
            # Per-test summaries
            for test_name in sorted(self.all_tests_data.keys()):
                test_data = self.all_tests_data[test_name]
                if test_data['bus_flips']:
                    df = pd.DataFrame(test_data['bus_flips'])
                    
                    # Create unit-station-save summary for this test
                    summary = df.groupby(['unit_id', 'station', 'save']).agg({
                        'msg_type': 'count',
                        'bus_transition': lambda x: ', '.join(sorted(set(x)))
                    }).reset_index()
                    summary.columns = ['unit_id', 'station', 'save', 'flip_count', 'transitions']
                    
                    # Truncate sheet name if too long
                    sheet_name = f"{test_name[:25]}_Summary" if len(test_name) > 25 else f"{test_name}_Summary"
                    summary.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\nSuper Excel saved to: {excel_path.absolute()}")
        return excel_path
    
    def create_super_dashboard(self):
        """Create a super dashboard comparing all tests"""
        dashboard_path = self.output_folder / "super_dashboard.html"
        
        # Prepare data for JavaScript
        test_comparison = self.df_test_comparison.to_dict('records') if self.df_test_comparison is not None else []
        all_flips = self.df_combined_flips.to_dict('records') if self.df_combined_flips is not None else []
        
        # Get unique values for filters
        test_names = sorted(self.df_combined_flips['test_name'].unique().tolist()) if self.df_combined_flips is not None else []
        unit_ids = sorted(self.df_combined_flips['unit_id'].unique().tolist()) if self.df_combined_flips is not None else []
        stations = sorted(self.df_combined_flips['station'].unique().tolist()) if self.df_combined_flips is not None else []
        saves = sorted(self.df_combined_flips['save'].unique().tolist()) if self.df_combined_flips is not None else []
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Test Bus Monitor Dashboard</title>
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
        .reset-btn {{
            background: #ff9800;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin: 20px 0;
        }}
        .reset-btn:hover {{
            background: #e68900;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Test Bus Monitor Dashboard</h1>
        
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{len(test_names)}</div>
                <div class="stat-label">Tests Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(all_flips):,}</div>
                <div class="stat-label">Total Bus Flips</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(unit_ids)}</div>
                <div class="stat-label">Unique Units</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(stations)}</div>
                <div class="stat-label">Unique Stations</div>
            </div>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label for="testFilter">Test Name:</label>
                <select id="testFilter" onchange="updateFilters()">
                    <option value="">All Tests</option>
                </select>
            </div>
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
        </div>
        
        <button class="reset-btn" onclick="resetFilters()">Reset All Filters</button>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Test</div>
            <div id="testChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Test Comparison - Flip Counts</div>
            <div id="comparisonChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">R vs T Distribution by Test</div>
            <div id="rtByTestChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Unit Performance Across Tests</div>
            <div id="unitChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Station Performance Across Tests</div>
            <div id="stationChart"></div>
        </div>
    </div>
    
    <script>
        // Data from Python
        const testComparison = {test_comparison};
        const allFlips = {all_flips};
        let filteredData = [...allFlips];
        
        // Unique values
        const testNames = {test_names};
        const unitIds = {unit_ids};
        const stations = {stations};
        const saves = {saves};
        
        // Initialize filters
        function initializeFilters() {{
            const testFilter = document.getElementById('testFilter');
            testNames.forEach(test => {{
                const option = document.createElement('option');
                option.value = test;
                option.textContent = test;
                testFilter.appendChild(option);
            }});
            
            const unitFilter = document.getElementById('unitFilter');
            unitIds.forEach(unit => {{
                const option = document.createElement('option');
                option.value = unit;
                option.textContent = unit;
                unitFilter.appendChild(option);
            }});
            
            const stationFilter = document.getElementById('stationFilter');
            stations.forEach(station => {{
                const option = document.createElement('option');
                option.value = station;
                option.textContent = station;
                stationFilter.appendChild(option);
            }});
            
            const saveFilter = document.getElementById('saveFilter');
            saves.forEach(save => {{
                const option = document.createElement('option');
                option.value = save;
                option.textContent = save;
                saveFilter.appendChild(option);
            }});
        }}
        
        function updateFilters() {{
            const testFilter = document.getElementById('testFilter').value;
            const unitFilter = document.getElementById('unitFilter').value;
            const stationFilter = document.getElementById('stationFilter').value;
            const saveFilter = document.getElementById('saveFilter').value;
            
            filteredData = allFlips.filter(row => {{
                return (!testFilter || row.test_name === testFilter) &&
                       (!unitFilter || row.unit_id === unitFilter) &&
                       (!stationFilter || row.station === stationFilter) &&
                       (!saveFilter || row.save === saveFilter);
            }});
            
            drawCharts();
        }}
        
        function resetFilters() {{
            document.getElementById('testFilter').value = '';
            document.getElementById('unitFilter').value = '';
            document.getElementById('stationFilter').value = '';
            document.getElementById('saveFilter').value = '';
            updateFilters();
        }}
        
        function drawCharts() {{
            // Test Chart
            const testCounts = {{}};
            filteredData.forEach(d => {{
                testCounts[d.test_name] = (testCounts[d.test_name] || 0) + 1;
            }});
            
            Plotly.newPlot('testChart', [{{
                x: Object.keys(testCounts),
                y: Object.values(testCounts),
                type: 'bar',
                marker: {{ color: '#667eea' }}
            }}], {{
                margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                xaxis: {{ title: 'Test Name', tickangle: -45 }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // Comparison Chart
            if (testComparison.length > 0) {{
                Plotly.newPlot('comparisonChart', [{{
                    x: testComparison.map(t => t.test_name),
                    y: testComparison.map(t => t.total_flips),
                    type: 'bar',
                    marker: {{ color: '#764ba2' }},
                    text: testComparison.map(t => t.total_flips),
                    textposition: 'auto'
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Test Name', tickangle: -45 }},
                    yaxis: {{ title: 'Total Flips' }}
                }});
                
                // R vs T by Test Chart
                const rData = testComparison.map(t => t.r_messages);
                const tData = testComparison.map(t => t.t_messages);
                
                Plotly.newPlot('rtByTestChart', [
                    {{
                        x: testComparison.map(t => t.test_name),
                        y: rData,
                        name: 'R Messages',
                        type: 'bar',
                        marker: {{ color: '#4CAF50' }}
                    }},
                    {{
                        x: testComparison.map(t => t.test_name),
                        y: tData,
                        name: 'T Messages',
                        type: 'bar',
                        marker: {{ color: '#2196F3' }}
                    }}
                ], {{
                    barmode: 'group',
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Test Name', tickangle: -45 }},
                    yaxis: {{ title: 'Count' }}
                }});
            }}
            
            // Unit Chart
            const unitCounts = {{}};
            filteredData.forEach(d => {{
                unitCounts[d.unit_id] = (unitCounts[d.unit_id] || 0) + 1;
            }});
            
            const sortedUnits = Object.entries(unitCounts).sort((a, b) => b[1] - a[1]).slice(0, 20);
            
            Plotly.newPlot('unitChart', [{{
                x: sortedUnits.map(x => x[0]),
                y: sortedUnits.map(x => x[1]),
                type: 'bar',
                marker: {{ color: '#FF9800' }}
            }}], {{
                margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                xaxis: {{ title: 'Unit ID', tickangle: -45 }},
                yaxis: {{ title: 'Number of Flips' }}
            }});
            
            // Station Chart
            const stationCounts = {{}};
            filteredData.forEach(d => {{
                stationCounts[d.station] = (stationCounts[d.station] || 0) + 1;
            }});
            
            const sortedStations = Object.entries(stationCounts).sort((a, b) => b[1] - a[1]).slice(0, 20);
            
            Plotly.newPlot('stationChart', [{{
                x: sortedStations.map(x => x[0]),
                y: sortedStations.map(x => x[1]),
                type: 'bar',
                marker: {{ color: '#9C27B0' }}
            }}], {{
                margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                xaxis: {{ title: 'Station', tickangle: -45 }},
                yaxis: {{ title: 'Number of Flips' }}
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
        
        print(f"Super dashboard saved to: {dashboard_path.absolute()}")
        return dashboard_path
    
    def print_summary(self):
        """Print summary of all tests"""
        print("\n" + "="*70)
        print("MULTI-TEST BUS MONITOR ANALYSIS SUMMARY")
        print("="*70)
        
        if self.df_test_comparison is not None and not self.df_test_comparison.empty:
            print(f"\nTests Analyzed: {len(self.df_test_comparison)}")
            print("-"*40)
            
            # Top 3 tests by flip count
            top_tests = self.df_test_comparison.head(3)
            print("\nTop 3 Tests by Bus Flips:")
            for idx, row in top_tests.iterrows():
                print(f"  {idx+1}. {row['test_name']}: {row['total_flips']} flips")
                print(f"     Files: {row['total_files']}, Units: {row['unique_units']}, Stations: {row['unique_stations']}")
                print(f"     R/T Distribution: {row['r_percentage']:.1f}% R, {row['t_percentage']:.1f}% T")
        
        print("\nOVERALL STATISTICS")
        print("-"*40)
        if self.df_combined_flips is not None:
            print(f"Total Bus Flips Across All Tests: {len(self.df_combined_flips):,}")
        if self.df_combined_summary is not None:
            print(f"Total Files Processed: {len(self.df_combined_summary):,}")
            print(f"Total Rows Analyzed: {self.df_combined_summary['total_rows'].sum():,}")
        if self.combined_header_issues:
            print(f"Total Header Issues: {len(self.combined_header_issues):,}")
        if self.combined_data_changes:
            print(f"Total Data Changes: {len(self.combined_data_changes):,}")
        
        print("\n" + "="*70)


def main():
    """Main execution function"""
    print("Starting Multi-Test Bus Monitor Analysis")
    print("-" * 60)
    
    analyzer = MultiTestBusMonitorDashboard()
    
    analyzer.run_analysis()
    
    if not analyzer.all_tests_data:
        print("\nNo data was processed. Please check:")
        print(f"  1. The parent folder path is correct: {analyzer.parent_folder}")
        print("  2. Test subfolders exist in that folder")
        print("  3. CSV files exist in the test subfolders")
        return
    
    analyzer.print_summary()
    
    print("\nExporting results...")
    print("-" * 60)
    
    # Export individual Excel files for each test
    print("\nExporting individual test files...")
    analyzer.export_individual_excels()
    
    # Export super Excel with all data
    print("\nExporting super Excel...")
    excel_path = analyzer.export_super_excel()
    
    # Create super dashboard
    print("\nCreating super dashboard...")
    dashboard_path = analyzer.create_super_dashboard()
    
    print("\n" + "="*70)
    print("MULTI-TEST ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {analyzer.output_folder.absolute()}")
    print("\nFiles Generated:")
    print(f"  Super Excel: super_analysis.xlsx (all tests combined)")
    print(f"  Super Dashboard: super_dashboard.html (interactive comparison)")
    print(f"  Individual Excel files: {len(analyzer.all_tests_data)} test files")
    print("\nOpen super_dashboard.html in a web browser for interactive visualizations!")


if __name__ == "__main__":
    main()
