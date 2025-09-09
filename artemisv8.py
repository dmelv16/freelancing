#!/usr/bin/env python3
"""
Bus Monitor Analysis Dashboard - Excel Export Version
Tracks bus flips with matching decoded_description values and exports to Excel
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BusMonitorDashboard:
    def __init__(self):
        """
        Initialize the Bus Monitor Dashboard
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
            
            # Create flip record
            flip_info = {
                'filename': file_info['filename'],
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
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
        Export all dataframes to Excel files
        """
        excel_path = self.output_folder / "bus_monitor_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Write each dataframe to a separate sheet
            if self.df_summary is not None and len(self.df_summary) > 0:
                self.df_summary.to_excel(writer, sheet_name='File Summary', index=False)
                print(f"  Exported File Summary: {len(self.df_summary)} rows")
            
            if self.df_file_messages is not None and len(self.df_file_messages) > 0:
                self.df_file_messages.to_excel(writer, sheet_name='Message Types', index=False)
                print(f"  Exported Message Types: {len(self.df_file_messages)} rows")
            
            if self.df_flips is not None and len(self.df_flips) > 0:
                self.df_flips.to_excel(writer, sheet_name='Flip Details', index=False)
                print(f"  Exported Flip Details: {len(self.df_flips)} rows")
            
            if self.df_changes is not None and len(self.df_changes) > 0:
                self.df_changes.to_excel(writer, sheet_name='Data Changes', index=False)
                print(f"  Exported Data Changes: {len(self.df_changes)} rows")
            
            if self.df_detailed_flips is not None and len(self.df_detailed_flips) > 0:
                # Truncate if too many columns for Excel
                if len(self.df_detailed_flips.columns) > 200:
                    self.df_detailed_flips.iloc[:, :200].to_excel(writer, sheet_name='Complete Flip Data', index=False)
                    print(f"  Exported Complete Flip Data: {len(self.df_detailed_flips)} rows (truncated to 200 columns)")
                else:
                    self.df_detailed_flips.to_excel(writer, sheet_name='Complete Flip Data', index=False)
                    print(f"  Exported Complete Flip Data: {len(self.df_detailed_flips)} rows")
            
            if self.df_headers is not None and len(self.df_headers) > 0:
                self.df_headers.to_excel(writer, sheet_name='Header Validation', index=False)
                print(f"  Exported Header Validation: {len(self.df_headers)} rows")
        
        print(f"\nExcel file saved to: {excel_path.absolute()}")
        return excel_path
    
    def export_individual_csvs(self):
        """
        Export each dataframe to individual CSV files
        """
        csv_files = []
        
        if self.df_summary is not None and len(self.df_summary) > 0:
            path = self.output_folder / "file_summary.csv"
            self.df_summary.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported file_summary.csv: {len(self.df_summary)} rows")
        
        if self.df_file_messages is not None and len(self.df_file_messages) > 0:
            path = self.output_folder / "message_types.csv"
            self.df_file_messages.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported message_types.csv: {len(self.df_file_messages)} rows")
        
        if self.df_flips is not None and len(self.df_flips) > 0:
            path = self.output_folder / "flip_details.csv"
            self.df_flips.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported flip_details.csv: {len(self.df_flips)} rows")
        
        if self.df_changes is not None and len(self.df_changes) > 0:
            path = self.output_folder / "data_changes.csv"
            self.df_changes.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported data_changes.csv: {len(self.df_changes)} rows")
        
        if self.df_detailed_flips is not None and len(self.df_detailed_flips) > 0:
            path = self.output_folder / "complete_flip_data.csv"
            self.df_detailed_flips.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported complete_flip_data.csv: {len(self.df_detailed_flips)} rows")
        
        if self.df_headers is not None and len(self.df_headers) > 0:
            path = self.output_folder / "header_validation.csv"
            self.df_headers.to_csv(path, index=False)
            csv_files.append(path)
            print(f"  Exported header_validation.csv: {len(self.df_headers)} rows")
        
        return csv_files
    
    def create_simple_html_report(self):
        """
        Create a simple HTML report with links to the Excel and CSV files
        """
        report_path = self.output_folder / "index.html"
        
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
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .stats {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    min-width: 150px;
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .stat-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .files {{
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 30px 0;
                }}
                .file-link {{
                    display: inline-block;
                    margin: 10px;
                    padding: 10px 20px;
                    background: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }}
                .file-link:hover {{
                    background: #45a049;
                }}
                .csv-link {{
                    background: #2196F3;
                }}
                .csv-link:hover {{
                    background: #0b7dda;
                }}
                .info {{
                    background: #fff3cd;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Bus Monitor Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis:</strong> Only tracking bus flips with matching decoded_description values</p>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{total_files}</div>
                    <div class="stat-label">Files Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_rows:,}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_flips}</div>
                    <div class="stat-label">Bus Flips</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_changes}</div>
                    <div class="stat-label">Data Changes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{header_issues}</div>
                    <div class="stat-label">Header Issues</div>
                </div>
            </div>
            
            <div class="files">
                <h2>Excel Report (All Data in One File)</h2>
                <a href="bus_monitor_analysis.xlsx" class="file-link">Download Excel Report</a>
                
                <h2>Individual CSV Files</h2>
                <a href="file_summary.csv" class="csv-link file-link">File Summary</a>
                <a href="message_types.csv" class="csv-link file-link">Message Types</a>
                <a href="flip_details.csv" class="csv-link file-link">Flip Details</a>
                <a href="data_changes.csv" class="csv-link file-link">Data Changes</a>
                <a href="complete_flip_data.csv" class="csv-link file-link">Complete Flip Data</a>
                <a href="header_validation.csv" class="csv-link file-link">Header Validation</a>
            </div>
            
            <div class="info">
                <h3>How to Use These Files:</h3>
                <ul>
                    <li><strong>Excel Report:</strong> Open in Microsoft Excel, Google Sheets, or LibreOffice Calc. Each analysis type is in a separate tab.</li>
                    <li><strong>CSV Files:</strong> Can be opened in any spreadsheet program or text editor. Best for importing into other analysis tools.</li>
                    <li><strong>File Summary:</strong> Overview of all processed files with flip statistics</li>
                    <li><strong>Message Types:</strong> All message types found in each file</li>
                    <li><strong>Flip Details:</strong> Every bus flip detected with timestamps and transitions</li>
                    <li><strong>Data Changes:</strong> Shows which data columns changed during flips</li>
                    <li><strong>Complete Flip Data:</strong> All columns for messages that flipped (Bus A vs Bus B)</li>
                    <li><strong>Header Validation:</strong> Mismatches between expected and actual headers</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nHTML index saved to: {report_path.absolute()}")
        return report_path
    
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
        
        if self.df_headers is not None and len(self.df_headers) > 0:
            print(f"\nHeader Validation Issues: {len(self.df_headers)}")
        
        if self.df_changes is not None and len(self.df_changes) > 0:
            print(f"\nTotal Data Word Changes: {len(self.df_changes)}")
        
        print("\n" + "="*60)


def main():
    """
    Main execution function
    """
    print("Starting Bus Monitor Analysis")
    print("-" * 60)
    
    analyzer = BusMonitorDashboard()
    
    results = analyzer.run_analysis()
    
    if not results:
        print("\nNo data was processed. Please check:")
        print(f"  1. The CSV folder path is correct: {analyzer.csv_folder}")
        print("  2. CSV files exist in that folder")
        print("  3. CSV files have the correct format")
        return
    
    analyzer.print_summary()
    
    print("\nExporting results...")
    print("-" * 60)
    
    # Export to Excel
    excel_path = analyzer.export_to_excel()
    
    # Export to individual CSVs
    print("\nExporting individual CSV files...")
    csv_files = analyzer.export_individual_csvs()
    
    # Create HTML index
    html_path = analyzer.create_simple_html_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {analyzer.output_folder.absolute()}")
    print("\nYou can:")
    print(f"1. Open the Excel file directly: {excel_path.name}")
    print(f"2. Open individual CSV files in Excel or any spreadsheet program")
    print(f"3. Open {html_path.name} in your browser for a summary with download links")
    print("\nThe Excel file contains all data in separate tabs for easy viewing.")


if __name__ == "__main__":
    main()
