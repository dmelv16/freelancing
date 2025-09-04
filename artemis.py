#!/usr/bin/env python3
"""
All-in-One Bus Monitor Validation System
No command line arguments needed - just configure the paths below and run!
"""

import pandas as pd
import numpy as np
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ============================================================================

# Input folders
CSV_FOLDER = r"C:\path\to\your\csv\logs"  # Folder with raw CSV logs
EXCEL_FOLDER = r"C:\path\to\your\excel\reports"  # Folder with Excel validation reports

# Output files (will be created in current directory)
DEFINITIONS_FILE = "learned_message_definitions.json"
VALIDATION_REPORT = "bus_monitor_validation_report.csv"
SUMMARY_REPORT = "validation_summary_by_type.csv"

# Options
LEARN_FROM_DATA = True  # Set to False to skip learning if definitions file already exists
SAMPLE_SIZE = None  # Set to a number (e.g., 10000) to limit rows per CSV for faster learning
VERBOSE = True  # Set to False for less output

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Setup logging
log_level = logging.DEBUG if VERBOSE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MessageError:
    """Represents an error from Excel report"""
    message_type: str
    data_word: str
    timestamp: float
    expected_values: List[str]
    found_values: List[str]
    unit_id: str
    station: str
    save: str

@dataclass
class ValidationResult:
    """Result of validation"""
    error: MessageError
    status: str
    error_bus: Optional[str]
    opposite_bus: Optional[str]
    correction_timestamp: Optional[float]
    all_data_words_checked: Dict = field(default_factory=dict)
    confidence: str = 'LOW'
    details: str = ''


class MessagePatternLearner:
    """Learn message patterns from Excel TRUE results and CSV logs"""
    
    def __init__(self):
        self.csv_folder = Path(CSV_FOLDER)
        self.excel_folder = Path(EXCEL_FOLDER)
        self.definitions_file = DEFINITIONS_FILE
        
        self.message_types = defaultdict(lambda: {
            'count': 0,
            'true_count': 0,
            'files_seen': set(),
            'true_sources': set(),
            'data_words': defaultdict(lambda: {
                'values': set(),
                'true_values': set(),
                'count': 0,
                'confidence': 'LOW'
            }),
            'base_type': None,
            'data01_identifier': None
        })
        
        self.total_rows = 0
        self.total_true_validations = 0
    
    def learn(self):
        """Main learning function"""
        logger.info("="*60)
        logger.info("STARTING PATTERN LEARNING")
        logger.info("="*60)
        
        # Learn from Excel TRUE results
        logger.info("Learning from Excel TRUE validations...")
        self.learn_from_excel()
        
        # Learn from CSV logs
        logger.info("Learning from CSV raw logs...")
        self.learn_from_csv()
        
        # Save definitions
        self.save_definitions()
        
        logger.info(f"Learning complete! Saved to {self.definitions_file}")
        return self.definitions_file
    
    def learn_from_excel(self):
        """Learn from Excel TRUE results"""
        excel_files = list(self.excel_folder.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files")
        
        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file)
                message_cols = [col for col in df.columns if re.match(r'\[.*\]', str(col))]
                
                for _, row in df.iterrows():
                    for msg_col in message_cols:
                        cell_value = str(row[msg_col])
                        
                        # Process TRUE results
                        if 'TRUE' in cell_value and 'MIXED' not in cell_value:
                            message_type = msg_col.strip('[]')
                            self.message_types[message_type]['true_count'] += 1
                            self.total_true_validations += 1
                            
                            # Get the CSV file for this TRUE result
                            unit_id = str(row.get('unit_id', ''))
                            station = str(row.get('station', ''))
                            save = str(row.get('save', ''))
                            
                            # Look up actual values
                            self.lookup_true_values(message_type, unit_id, station, save)
                            
            except Exception as e:
                logger.debug(f"Error processing {excel_file.name}: {e}")
        
        logger.info(f"Learned from {self.total_true_validations} TRUE validations")
    
    def lookup_true_values(self, message_type: str, unit_id: str, station: str, save: str):
        """Look up actual values from CSV for TRUE validation"""
        # Build CSV filename
        bus_type = 'rt' if station.startswith('R') else 'lt'
        station_num = re.search(r'\d+', station)
        suffix = (station_num.group() if station_num else '01').zfill(2)
        
        csv_file = self.csv_folder / f"{unit_id}_{station}_{save}_{bus_type}{suffix}.csv"
        
        if not csv_file.exists():
            return
        
        try:
            # Sample the CSV to find examples of this message type
            df = pd.read_csv(csv_file, nrows=1000)
            
            if 'message_name' in df.columns:
                msg_rows = df[df['message_name'] == message_type]
                
                for _, row in msg_rows.iterrows():
                    # Learn all data word values
                    for col in df.columns:
                        if col.startswith('data') and pd.notna(row[col]):
                            value = str(row[col])
                            self.message_types[message_type]['data_words'][col]['values'].add(value)
                            self.message_types[message_type]['data_words'][col]['true_values'].add(value)
                            self.message_types[message_type]['data_words'][col]['count'] += 1
                    
                    # Track base type
                    if 'decoded description' in row:
                        match = re.search(r'\[([^\]]+)\]', str(row['decoded description']))
                        if match:
                            self.message_types[message_type]['base_type'] = match.group(1)
                    
                    # Track data01 identifier
                    if 'data01' in row and pd.notna(row['data01']):
                        self.message_types[message_type]['data01_identifier'] = str(row['data01'])
                        
        except Exception as e:
            logger.debug(f"Error reading {csv_file.name}: {e}")
    
    def learn_from_csv(self):
        """Learn patterns from CSV files"""
        csv_files = list(self.csv_folder.glob("*.csv"))[:20]  # Sample first 20 files
        logger.info(f"Sampling {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=SAMPLE_SIZE)
                
                if 'message_name' not in df.columns:
                    continue
                
                for _, row in df.iterrows():
                    self.total_rows += 1
                    
                    if pd.notna(row['message_name']):
                        message_type = str(row['message_name'])
                        
                        # Skip error cases where message_name = decoded_description
                        if 'decoded description' in row and message_type == str(row['decoded description']):
                            continue
                        
                        self.message_types[message_type]['count'] += 1
                        
                        # Learn data words
                        for col in df.columns:
                            if col.startswith('data') and pd.notna(row[col]):
                                value = str(row[col])
                                self.message_types[message_type]['data_words'][col]['values'].add(value)
                                self.message_types[message_type]['data_words'][col]['count'] += 1
                
            except Exception as e:
                logger.debug(f"Error processing {csv_file.name}: {e}")
        
        logger.info(f"Processed {self.total_rows:,} rows")
    
    def save_definitions(self):
        """Save learned definitions to JSON"""
        definitions = {}
        
        for msg_type, info in self.message_types.items():
            # Determine confidence for each data word
            for dw, dw_info in info['data_words'].items():
                if dw_info['true_values']:
                    dw_info['confidence'] = 'HIGH'
                elif dw_info['count'] > 100:
                    dw_info['confidence'] = 'MEDIUM'
            
            definitions[msg_type] = {
                'message_type': msg_type,
                'base_type': info['base_type'],
                'occurrences': info['count'],
                'true_validations': info['true_count'],
                'data01_identifier': info['data01_identifier'],
                'data_words': {}
            }
            
            # Process data words
            for dw, dw_info in info['data_words'].items():
                values = list(dw_info['true_values'] if dw_info['true_values'] else dw_info['values'])
                
                if len(values) == 1:
                    definitions[msg_type]['data_words'][dw] = {
                        'type': 'single_value',
                        'value': values[0],
                        'confidence': dw_info['confidence']
                    }
                elif len(values) <= 50:
                    definitions[msg_type]['data_words'][dw] = {
                        'type': 'multiple_values',
                        'values': sorted(values),
                        'confidence': dw_info['confidence']
                    }
                else:
                    definitions[msg_type]['data_words'][dw] = {
                        'type': 'dynamic',
                        'sample_values': sorted(values)[:20],
                        'confidence': dw_info['confidence']
                    }
        
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_rows': self.total_rows,
                'total_true_validations': self.total_true_validations,
                'unique_message_types': len(definitions)
            },
            'message_definitions': definitions
        }
        
        with open(self.definitions_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(definitions)} message type definitions")


class BusMonitorValidator:
    """Validate bus monitor errors using learned definitions"""
    
    def __init__(self, definitions_file: str):
        self.csv_folder = Path(CSV_FOLDER)
        self.excel_folder = Path(EXCEL_FOLDER)
        self.results = []
        
        # Load definitions
        with open(definitions_file, 'r') as f:
            data = json.load(f)
        
        self.definitions = data.get('message_definitions', {})
        self.metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(self.definitions)} message definitions")
    
    def validate_all(self):
        """Process all Excel reports"""
        logger.info("="*60)
        logger.info("STARTING VALIDATION")
        logger.info("="*60)
        
        excel_files = list(self.excel_folder.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel reports to validate")
        
        for excel_file in excel_files:
            logger.info(f"Processing: {excel_file.name}")
            self.process_excel(excel_file)
        
        # Generate reports
        self.generate_reports()
        
        # Print summary
        self.print_summary()
    
    def process_excel(self, excel_file: Path):
        """Process one Excel file"""
        try:
            df = pd.read_excel(excel_file)
            message_cols = [col for col in df.columns if re.match(r'\[.*\]', str(col))]
            
            for _, row in df.iterrows():
                for msg_col in message_cols:
                    cell_value = str(row[msg_col])
                    
                    # Process MIXED or FALSE
                    if ('MIXED' in cell_value or 'FALSE' in cell_value) and 'TRUE' not in cell_value:
                        message_type = msg_col.strip('[]')
                        errors = self.parse_errors(
                            cell_value, message_type,
                            str(row.get('unit_id', '')),
                            str(row.get('station', '')),
                            str(row.get('save', ''))
                        )
                        
                        for error in errors:
                            result = self.validate_error(error)
                            self.results.append(result)
                            
        except Exception as e:
            logger.error(f"Error processing {excel_file.name}: {e}")
    
    def parse_errors(self, cell_text: str, message_type: str,
                    unit_id: str, station: str, save: str) -> List[MessageError]:
        """Parse errors from cell text"""
        errors = []
        
        # Find all data word errors
        pattern = r'messages had incorrect format for (data\d+)\.\s*Expected any of \[(.*?)\].*?found \[(.*?)\]\.\s*Fails occurred at timestamps \[([\d.,\s]+)\]'
        
        for match in re.finditer(pattern, cell_text, re.DOTALL):
            data_word = match.group(1)
            expected = [v.strip().strip("'\"") for v in match.group(2).split(',')]
            found = [v.strip().strip("'\"") for v in match.group(3).split(',')]
            timestamps = [float(t.strip()) for t in match.group(4).split(',')]
            
            for timestamp in timestamps:
                errors.append(MessageError(
                    message_type=message_type,
                    data_word=data_word,
                    timestamp=timestamp,
                    expected_values=expected,
                    found_values=found,
                    unit_id=unit_id,
                    station=station,
                    save=save
                ))
        
        return errors
    
    def validate_error(self, error: MessageError) -> ValidationResult:
        """Validate a single error"""
        # Get CSV file
        bus_type = 'rt' if error.station.startswith('R') else 'lt'
        station_num = re.search(r'\d+', error.station)
        suffix = (station_num.group() if station_num else '01').zfill(2)
        csv_file = self.csv_folder / f"{error.unit_id}_{error.station}_{error.save}_{bus_type}{suffix}.csv"
        
        if not csv_file.exists():
            return ValidationResult(
                error=error,
                status="FILE_NOT_FOUND",
                error_bus=None,
                opposite_bus=None,
                correction_timestamp=None,
                details=f"CSV not found: {csv_file.name}"
            )
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Find error in time window
            time_window = 0.01
            window = df[(df['timestamp'] >= error.timestamp - time_window) & 
                       (df['timestamp'] <= error.timestamp + time_window)].copy()
            
            if window.empty:
                return ValidationResult(
                    error=error,
                    status="NO_DATA",
                    error_bus=None,
                    opposite_bus=None,
                    correction_timestamp=None,
                    details="No data in time window"
                )
            
            # Find error message
            error_bus = None
            for _, row in window.iterrows():
                if abs(row['timestamp'] - error.timestamp) < 0.001:
                    if error.data_word in df.columns:
                        if str(row[error.data_word]) in error.found_values:
                            error_bus = row['bus']
                            break
            
            if not error_bus:
                return ValidationResult(
                    error=error,
                    status="ERROR_NOT_FOUND",
                    error_bus=None,
                    opposite_bus=None,
                    correction_timestamp=None,
                    details="Could not find error"
                )
            
            # Check opposite bus for correction
            opposite_bus = 'B' if error_bus == 'A' else 'A'
            
            # Look for correction
            correction_found = False
            correction_time = None
            
            opposite_msgs = window[window['bus'] == opposite_bus]
            for _, row in opposite_msgs.iterrows():
                if error.data_word in df.columns:
                    if str(row[error.data_word]) in error.expected_values:
                        correction_found = True
                        correction_time = row['timestamp']
                        break
            
            if correction_found:
                return ValidationResult(
                    error=error,
                    status="BUS_MONITOR_ERROR",
                    error_bus=error_bus,
                    opposite_bus=opposite_bus,
                    correction_timestamp=correction_time,
                    details=f"Corrected on Bus {opposite_bus}"
                )
            else:
                return ValidationResult(
                    error=error,
                    status="TRUE_FAILURE",
                    error_bus=error_bus,
                    opposite_bus=None,
                    correction_timestamp=None,
                    details="No correction found"
                )
                
        except Exception as e:
            return ValidationResult(
                error=error,
                status="ERROR",
                error_bus=None,
                opposite_bus=None,
                correction_timestamp=None,
                details=str(e)
            )
    
    def generate_reports(self):
        """Generate output reports"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Main report
        report_data = []
        for r in self.results:
            report_data.append({
                'Unit_ID': r.error.unit_id,
                'Station': r.error.station,
                'Save': r.error.save,
                'Message_Type': r.error.message_type,
                'Data_Word': r.error.data_word,
                'Timestamp': r.error.timestamp,
                'Status': r.status,
                'Error_Bus': r.error_bus,
                'Correction_Bus': r.opposite_bus,
                'Correction_Time': r.correction_timestamp,
                'Details': r.details
            })
        
        df = pd.DataFrame(report_data)
        df.to_csv(VALIDATION_REPORT, index=False)
        logger.info(f"Saved main report to: {VALIDATION_REPORT}")
        
        # Summary by message type
        summary = df.groupby(['Message_Type', 'Status']).size().unstack(fill_value=0)
        summary.to_csv(SUMMARY_REPORT)
        logger.info(f"Saved summary to: {SUMMARY_REPORT}")
    
    def print_summary(self):
        """Print summary statistics"""
        bus_monitor = sum(1 for r in self.results if r.status == "BUS_MONITOR_ERROR")
        true_failure = sum(1 for r in self.results if r.status == "TRUE_FAILURE")
        total = bus_monitor + true_failure
        
        logger.info("\n" + "="*60)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*60)
        
        if total > 0:
            rate = (bus_monitor / total) * 100
            logger.info(f"Bus Monitor Errors: {bus_monitor} ({rate:.1f}%)")
            logger.info(f"True Failures: {true_failure} ({100-rate:.1f}%)")
        
        logger.info(f"Total Errors Analyzed: {len(self.results)}")


def main():
    """Main function - no arguments needed!"""
    
    print("\n" + "="*70)
    print("BUS MONITOR VALIDATION SYSTEM")
    print("="*70)
    print(f"\nCSV Folder: {CSV_FOLDER}")
    print(f"Excel Folder: {EXCEL_FOLDER}")
    print(f"Output Files: {DEFINITIONS_FILE}, {VALIDATION_REPORT}\n")
    
    # Step 1: Learn patterns (if needed)
    if LEARN_FROM_DATA or not os.path.exists(DEFINITIONS_FILE):
        print("\nStep 1: Learning message patterns...")
        learner = MessagePatternLearner()
        definitions_file = learner.learn()
    else:
        print(f"\nStep 1: Using existing definitions from {DEFINITIONS_FILE}")
        definitions_file = DEFINITIONS_FILE
    
    # Step 2: Validate errors
    print("\nStep 2: Validating bus monitor errors...")
    validator = BusMonitorValidator(definitions_file)
    validator.validate_all()
    
    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nCheck these files for results:")
    print(f"  - {VALIDATION_REPORT}")
    print(f"  - {SUMMARY_REPORT}")
    print(f"  - {DEFINITIONS_FILE}")


if __name__ == "__main__":
    main()
