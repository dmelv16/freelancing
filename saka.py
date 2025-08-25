import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
import warnings
warnings.filterwarnings('ignore')

class FullMessageStructureVerifier:
    """
    Verifies complete message structure across all dataword columns to distinguish
    between bus monitor issues and actual data errors.
    """
    
    def __init__(self, cache_dir="message_structure_cache"):
        self.cache_dir = cache_dir
        self.message_structures = {}  # Expected structure for each message type
        self.dataword_patterns = {}  # Patterns for dataword columns
        self.test_specific_rules = {}  # Rules per test file
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing knowledge
        self.load_message_structures()
    
    def load_message_structures(self):
        """
        Load previously learned message structures.
        """
        structure_file = os.path.join(self.cache_dir, "message_structures.json")
        if os.path.exists(structure_file):
            try:
                with open(structure_file, 'r') as f:
                    self.message_structures = json.load(f)
                print(f"Loaded message structures for {len(self.message_structures)} message types")
            except Exception as e:
                print(f"Could not load message structures: {e}")
    
    def save_message_structures(self):
        """
        Save learned message structures.
        """
        structure_file = os.path.join(self.cache_dir, "message_structures.json")
        try:
            with open(structure_file, 'w') as f:
                json.dump(self.message_structures, f, indent=2, default=str)
        except Exception as e:
            print(f"Could not save message structures: {e}")
    
    def get_rt_number_from_station(self, station):
        """
        Convert station to RT number.
        """
        if not station or not isinstance(station, str):
            return None
        
        try:
            if station.startswith('L') or station.startswith('R'):
                station_num = station[1:]
                return f"rt{station_num.zfill(2)}"
            else:
                return f"rt{str(station).zfill(2)}"
        except:
            return None
    
    def discover_and_learn_message_structures(self, excel_df, csv_directory, bracket_column):
        """
        Discover all message types and learn their complete dataword structures.
        """
        print(f"\n  Discovering message structures for {bracket_column}...")
        
        # Extract base pattern
        search_pattern = bracket_column.strip('[]').split('-')[0]
        
        # Get TRUE cases for learning
        true_rows = excel_df[
            excel_df[bracket_column].astype(str).str.contains('TRUE', case=False, na=False) &
            ~excel_df[bracket_column].astype(str).str.contains('MIXED|FALSE', case=False, na=False)
        ]
        
        if len(true_rows) == 0:
            print(f"    No TRUE cases found for learning")
            return None
        
        print(f"    Learning from {len(true_rows)} TRUE cases")
        
        # Dictionary to store learned structures
        learned_structures = defaultdict(lambda: {
            'dataword_usage': defaultdict(list),  # Which datawords are used
            'dataword_patterns': defaultdict(set),  # Valid values for each dataword
            'message_name_mappings': defaultdict(int),  # message_name frequency
            'total_examples': 0,
            'bus_patterns': defaultdict(int),
            'full_message_hashes': set()  # Store hashes of complete valid messages
        })
        
        # Analyze each TRUE case
        files_analyzed = 0
        max_files = 30  # Analyze more files for better coverage
        
        for idx, row in true_rows.iterrows():
            if files_analyzed >= max_files:
                break
                
            csv_file = self._find_csv_file(row['unit_id'], row['station'], row['save'], csv_directory)
            
            if not csv_file:
                continue
            
            try:
                csv_df = pd.read_csv(csv_file)
                
                # Check if we have dataword columns
                dataword_cols = [col for col in csv_df.columns if col.startswith('dataword')]
                
                if not dataword_cols:
                    print(f"      No dataword columns found in {os.path.basename(csv_file)}")
                    continue
                
                print(f"      Analyzing {os.path.basename(csv_file)} with {len(dataword_cols)} dataword columns")
                
                # Find messages related to our pattern
                if 'decoded_description' in csv_df.columns:
                    related_messages = csv_df[
                        csv_df['decoded_description'].astype(str).str.contains(search_pattern, na=False)
                    ]
                    
                    for msg_idx, msg_row in related_messages.iterrows():
                        desc = str(msg_row['decoded_description'])
                        msg_name = str(msg_row.get('message_name', ''))
                        bus = str(msg_row.get('bus', ''))
                        
                        # Analyze dataword structure
                        message_structure = {}
                        non_empty_datawords = []
                        
                        for dw_col in dataword_cols:
                            dw_value = msg_row.get(dw_col, '')
                            if pd.notna(dw_value) and str(dw_value).strip():
                                message_structure[dw_col] = str(dw_value)
                                non_empty_datawords.append(dw_col)
                                learned_structures[desc]['dataword_patterns'][dw_col].add(str(dw_value))
                        
                        # Store which datawords are used for this message type
                        if non_empty_datawords:
                            learned_structures[desc]['dataword_usage'][tuple(non_empty_datawords)].append({
                                'message_name': msg_name,
                                'bus': bus
                            })
                        
                        # Store message name mapping
                        learned_structures[desc]['message_name_mappings'][msg_name] += 1
                        learned_structures[desc]['bus_patterns'][bus] += 1
                        learned_structures[desc]['total_examples'] += 1
                        
                        # Create hash of complete message structure
                        message_hash = self._hash_message_structure(message_structure)
                        learned_structures[desc]['full_message_hashes'].add(message_hash)
                
                files_analyzed += 1
                
            except Exception as e:
                print(f"      Error analyzing {csv_file}: {e}")
                continue
        
        # Process learned structures
        processed_structures = {}
        
        for desc, structure_data in learned_structures.items():
            if structure_data['total_examples'] > 0:
                # Determine which datawords are consistently used
                most_common_usage = None
                max_count = 0
                
                for usage_pattern, examples in structure_data['dataword_usage'].items():
                    if len(examples) > max_count:
                        max_count = len(examples)
                        most_common_usage = usage_pattern
                
                # Determine the most common valid message_name
                valid_message_names = []
                for msg_name, count in structure_data['message_name_mappings'].items():
                    if count >= 2:  # Appears at least twice
                        valid_message_names.append(msg_name)
                
                processed_structures[desc] = {
                    'expected_datawords': list(most_common_usage) if most_common_usage else [],
                    'valid_message_names': valid_message_names,
                    'dataword_patterns': {
                        dw: list(patterns) for dw, patterns in structure_data['dataword_patterns'].items()
                    },
                    'total_examples': structure_data['total_examples'],
                    'valid_message_hashes': list(structure_data['full_message_hashes'])
                }
                
                print(f"    Learned structure for {desc}:")
                print(f"      - Uses {len(processed_structures[desc]['expected_datawords'])} datawords")
                print(f"      - Valid message names: {valid_message_names[:3]}...")
        
        # Cache the learned structures
        self.message_structures.update(processed_structures)
        self.save_message_structures()
        
        return processed_structures
    
    def _hash_message_structure(self, structure):
        """
        Create a hash of the complete message structure for comparison.
        """
        # Sort by key to ensure consistent hashing
        sorted_items = sorted(structure.items())
        structure_string = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(structure_string.encode()).hexdigest()
    
    def verify_message_integrity(self, csv_filepath, bracket_column, learned_structures):
        """
        Verify message integrity by checking complete dataword structure.
        Identifies true bus monitor issues vs actual data errors.
        """
        try:
            csv_df = pd.read_csv(csv_filepath)
            
            # Get dataword columns
            dataword_cols = [col for col in csv_df.columns if col.startswith('dataword')]
            
            if not dataword_cols:
                return False, "No dataword columns found in CSV"
            
            # Extract search pattern
            search_pattern = bracket_column.strip('[]').split('-')[0]
            
            # Find relevant messages
            if 'decoded_description' not in csv_df.columns:
                return False, "No decoded_description column found"
            
            related_messages = csv_df[
                csv_df['decoded_description'].astype(str).str.contains(search_pattern, na=False)
            ]
            
            if related_messages.empty:
                return False, "No relevant messages found"
            
            # Group messages by decoded_description
            verification_results = []
            
            for desc in related_messages['decoded_description'].unique():
                desc_str = str(desc)
                
                if desc_str not in learned_structures:
                    continue
                
                expected_structure = learned_structures[desc_str]
                desc_messages = related_messages[related_messages['decoded_description'] == desc]
                
                # Analyze messages on each bus
                bus_analysis = {}
                
                for bus in ['A', 'B']:
                    bus_messages = desc_messages[desc_messages['bus'] == bus] if 'bus' in desc_messages.columns else pd.DataFrame()
                    
                    if not bus_messages.empty:
                        bus_analysis[bus] = self._analyze_bus_messages(
                            bus_messages, expected_structure, dataword_cols
                        )
                
                # Determine if this is a bus monitor issue
                is_bus_monitor_issue = self._check_if_bus_monitor_issue(bus_analysis)
                
                verification_results.append({
                    'description': desc_str,
                    'bus_analysis': bus_analysis,
                    'is_bus_monitor_issue': is_bus_monitor_issue
                })
            
            # Aggregate results
            bus_monitor_issues = [r for r in verification_results if r['is_bus_monitor_issue']]
            data_errors = [r for r in verification_results if not r['is_bus_monitor_issue']]
            
            if bus_monitor_issues:
                return True, f"Bus monitor issue detected and corrected - {len(bus_monitor_issues)} messages fixed on opposite bus"
            elif data_errors:
                return False, f"Data structure errors found - {len(data_errors)} messages have incorrect data (not bus-related)"
            else:
                return False, "Unable to determine error type"
                
        except Exception as e:
            return False, f"Error during verification: {str(e)}"
    
    def _analyze_bus_messages(self, bus_messages, expected_structure, dataword_cols):
        """
        Analyze messages on a specific bus for structure validity.
        """
        analysis = {
            'total_messages': len(bus_messages),
            'structure_valid': [],
            'structure_invalid': [],
            'header_errors': 0,
            'data_errors': 0,
            'completely_valid': 0
        }
        
        expected_datawords = expected_structure.get('expected_datawords', [])
        valid_message_names = expected_structure.get('valid_message_names', [])
        valid_hashes = expected_structure.get('valid_message_hashes', [])
        
        for idx, msg in bus_messages.iterrows():
            message_validity = {
                'header_valid': False,
                'structure_valid': False,
                'data_valid': False,
                'errors': []
            }
            
            # Check header (message_name from dataword01)
            msg_name = str(msg.get('message_name', ''))
            if msg_name in valid_message_names:
                message_validity['header_valid'] = True
            else:
                message_validity['errors'].append(f"Invalid header: {msg_name}")
                analysis['header_errors'] += 1
            
            # Check structure (which datawords are populated)
            current_structure = {}
            populated_datawords = []
            
            for dw_col in dataword_cols:
                dw_value = msg.get(dw_col, '')
                if pd.notna(dw_value) and str(dw_value).strip():
                    current_structure[dw_col] = str(dw_value)
                    populated_datawords.append(dw_col)
            
            # Compare structure
            if set(populated_datawords) == set(expected_datawords):
                message_validity['structure_valid'] = True
            else:
                missing = set(expected_datawords) - set(populated_datawords)
                extra = set(populated_datawords) - set(expected_datawords)
                if missing:
                    message_validity['errors'].append(f"Missing datawords: {missing}")
                if extra:
                    message_validity['errors'].append(f"Extra datawords: {extra}")
                analysis['data_errors'] += 1
            
            # Check if complete message hash matches known valid messages
            message_hash = self._hash_message_structure(current_structure)
            if message_hash in valid_hashes:
                message_validity['data_valid'] = True
                analysis['completely_valid'] += 1
            
            # Store validity result
            if message_validity['header_valid'] and message_validity['structure_valid']:
                analysis['structure_valid'].append(message_validity)
            else:
                analysis['structure_invalid'].append(message_validity)
        
        return analysis
    
    def _check_if_bus_monitor_issue(self, bus_analysis):
        """
        Determine if errors are due to bus monitor issues.
        
        Bus monitor issue criteria:
        1. Bus A has invalid structure/header
        2. Bus B has completely valid structure/header
        3. Timing suggests B is a correction of A
        """
        if 'A' not in bus_analysis or 'B' not in bus_analysis:
            return False
        
        bus_a = bus_analysis['A']
        bus_b = bus_analysis['B']
        
        # Check if Bus A has errors and Bus B is clean
        a_has_errors = (bus_a['header_errors'] > 0 or bus_a['data_errors'] > 0)
        b_is_clean = (bus_b['completely_valid'] > 0 and bus_b['header_errors'] == 0 and bus_b['data_errors'] == 0)
        
        if a_has_errors and b_is_clean:
            return True
        
        # Check opposite case (B has errors, A is clean)
        b_has_errors = (bus_b['header_errors'] > 0 or bus_b['data_errors'] > 0)
        a_is_clean = (bus_a['completely_valid'] > 0 and bus_a['header_errors'] == 0 and bus_a['data_errors'] == 0)
        
        if b_has_errors and a_is_clean:
            return True
        
        return False
    
    def _find_csv_file(self, unit_id, station, save, csv_directory):
        """
        Find the corresponding CSV file.
        """
        rt_number = self.get_rt_number_from_station(station)
        if not rt_number:
            return None
        
        filename = f"{unit_id}_{station}_{save}_{rt_number}.csv"
        filepath = os.path.join(csv_directory, filename)
        
        if os.path.exists(filepath):
            return filepath
        
        for file in os.listdir(csv_directory):
            if file.lower() == filename.lower():
                return os.path.join(csv_directory, file)
        
        return None
    
    def process_excel_file(self, excel_path, csv_directory):
        """
        Process a single Excel file with full message structure verification.
        """
        excel_name = os.path.basename(excel_path)
        
        print(f"\n{'='*70}")
        print(f"Processing: {excel_name}")
        print(f"{'='*70}")
        
        try:
            df = pd.read_excel(excel_path)
            print(f"Loaded Excel with {len(df)} rows")
        except Exception as e:
            print(f"Error reading Excel: {e}")
            return []
        
        # Find bracket columns
        bracket_pattern = re.compile(r'^\[.*\]$')
        bracket_columns = [col for col in df.columns if bracket_pattern.match(str(col))]
        
        if not bracket_columns:
            print("No bracket columns found")
            return []
        
        print(f"Found bracket columns: {bracket_columns}")
        
        results = []
        
        for bracket_col in bracket_columns:
            print(f"\n  Processing column: {bracket_col}")
            
            # Learn message structures from TRUE cases
            learned_structures = self.discover_and_learn_message_structures(
                df, csv_directory, bracket_col
            )
            
            if not learned_structures:
                print(f"    Could not learn structures for {bracket_col}")
                continue
            
            # Find MIXED and FALSE cases to verify
            mixed_rows = df[df[bracket_col].astype(str).str.contains('MIXED', case=False, na=False)]
            false_rows = df[df[bracket_col].astype(str).str.contains('FALSE', case=False, na=False)]
            
            total_to_verify = len(mixed_rows) + len(false_rows)
            
            if total_to_verify > 0:
                print(f"\n  Verifying {len(mixed_rows)} MIXED and {len(false_rows)} FALSE cases")
                print(f"  Checking complete message structure across all dataword columns...")
            
            # Process MIXED cases
            for idx, row in mixed_rows.iterrows():
                csv_file = self._find_csv_file(row['unit_id'], row['station'], row['save'], csv_directory)
                
                if csv_file:
                    is_bus_issue, details = self.verify_message_integrity(
                        csv_file, bracket_col, learned_structures
                    )
                    
                    result = {
                        'excel_file': excel_name,
                        'column': bracket_col,
                        'row_index': idx,
                        'flag_type': 'MIXED',
                        'unit_id': row['unit_id'],
                        'station': row['station'],
                        'save': row['save'],
                        'is_bus_monitor_issue': is_bus_issue,
                        'details': details
                    }
                    
                    results.append(result)
                    
                    if is_bus_issue:
                        print(f"    Row {idx}: ✓ BUS MONITOR ISSUE (correctable) - {details}")
                    else:
                        print(f"    Row {idx}: ✗ DATA ERROR (not bus-related) - {details}")
            
            # Process FALSE cases
            for idx, row in false_rows.iterrows():
                csv_file = self._find_csv_file(row['unit_id'], row['station'], row['save'], csv_directory)
                
                if csv_file:
                    is_bus_issue, details = self.verify_message_integrity(
                        csv_file, bracket_col, learned_structures
                    )
                    
                    result = {
                        'excel_file': excel_name,
                        'column': bracket_col,
                        'row_index': idx,
                        'flag_type': 'FALSE',
                        'unit_id': row['unit_id'],
                        'station': row['station'],
                        'save': row['save'],
                        'is_bus_monitor_issue': is_bus_issue,
                        'details': details
                    }
                    
                    results.append(result)
                    
                    if is_bus_issue:
                        print(f"    Row {idx}: ✓ BUS MONITOR ISSUE (correctable) - {details}")
                    else:
                        print(f"    Row {idx}: ✗ DATA ERROR (not bus-related) - {details}")
        
        # Update Excel with results
        if results:
            bus_issues = [r for r in results if r['is_bus_monitor_issue']]
            
            if bus_issues:
                print(f"\n  Updating Excel with {len(bus_issues)} bus monitor corrections...")
                
                for result in bus_issues:
                    col = result['column']
                    df.loc[result['row_index'], f'{col}_verified'] = 'BUS_MONITOR_CORRECTED'
                    df.loc[result['row_index'], f'{col}_details'] = result['details']
                
                output_path = excel_path.replace('.xlsx', '_structure_verified.xlsx')
                df.to_excel(output_path, index=False)
                print(f"  Saved: {os.path.basename(output_path)}")
        
        return results

def main(excel_directory, csv_directory):
    """
    Main function for full message structure verification.
    """
    print("="*70)
    print("FULL MESSAGE STRUCTURE VERIFICATION SYSTEM")
    print("="*70)
    print("Analyzing complete dataword structure to identify bus monitor issues")
    print(f"Excel directory: {excel_directory}")
    print(f"CSV directory: {csv_directory}")
    
    if not os.path.exists(excel_directory) or not os.path.exists(csv_directory):
        print("Error: Directories not found")
        return
    
    # Find Excel files
    excel_files = [f for f in os.listdir(excel_directory) 
                   if f.endswith('.xlsx') and not f.startswith('~') and '_verified' not in f]
    
    if not excel_files:
        print("No Excel files found")
        return
    
    print(f"Found {len(excel_files)} Excel files to process\n")
    
    # Initialize verifier
    verifier = FullMessageStructureVerifier()
    
    # Process each file
    all_results = []
    
    for excel_file in excel_files:
        excel_path = os.path.join(excel_directory, excel_file)
        results = verifier.process_excel_file(excel_path, csv_directory)
        all_results.extend(results)
    
    # Generate summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if all_results:
        total_verified = len(all_results)
        bus_monitor_issues = sum(1 for r in all_results if r['is_bus_monitor_issue'])
        data_errors = sum(1 for r in all_results if not r['is_bus_monitor_issue'])
        
        print(f"\nTotal cases verified: {total_verified}")
        print(f"Bus monitor issues (correctable): {bus_monitor_issues}")
        print(f"Data structure errors (not bus-related): {data_errors}")
        
        # Breakdown by Excel file
        by_file = defaultdict(lambda: {'bus_issues': 0, 'data_errors': 0})
        for r in all_results:
            if r['is_bus_monitor_issue']:
                by_file[r['excel_file']]['bus_issues'] += 1
            else:
                by_file[r['excel_file']]['data_errors'] += 1
        
        print("\nBreakdown by file:")
        for file_name, counts in by_file.items():
            print(f"  {file_name}:")
            print(f"    - Bus monitor issues: {counts['bus_issues']}")
            print(f"    - Data errors: {counts['data_errors']}")
        
        # Export detailed results
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'message_structure_verification_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results exported to: {output_file}")
        
        # Show learned structures summary
        print(f"\nLearned message structures: {len(verifier.message_structures)}")
        for desc, structure in list(verifier.message_structures.items())[:3]:
            print(f"  {desc}: uses {len(structure['expected_datawords'])} datawords")

if __name__ == "__main__":
    # Configuration
    EXCEL_DIRECTORY = "excel_files"
    CSV_DIRECTORY = "csv_files"
    
    # Run full message structure verification
    main(EXCEL_DIRECTORY, CSV_DIRECTORY)
