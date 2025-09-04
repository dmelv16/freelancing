#!/usr/bin/env python3
"""
All-in-One Bus Monitor Validation System
Correctly tracks message types from decoded description with data01 subtypes
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
    message_type: str  # From Excel column like [190r-2]
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
    """Learn message patterns correctly from decoded description + data01"""
    
    def __init__(self):
        self.csv_folder = Path(CSV_FOLDER)
        self.excel_folder = Path(EXCEL_FOLDER)
        self.definitions_file = DEFINITIONS_FILE
        
        # Structure: {excel_type: {data01_value: message_info}}
        # e.g., {"190r-1": {"0x01": {...}}, "190r-2": {"0x02": {...}}}
        self.message_types = defaultdict(lambda: {
            'base_type': None,  # From decoded description (e.g., "190R")
            'excel_type': None,  # From Excel column (e.g., "190r-2")
            'data01_value': None,  # The data01 that identifies this subtype
            'message_name': None,  # Human readable name (e.g., "STORE")
            'count': 0,
            'true_count': 0,
            'true_timestamps': [],
            'files_seen': set(),
            'true_sources': set(),
            'data_words': defaultdict(lambda: {
                'values': set(),
                'true_values': set(),
                'count': 0,
                'confidence': 'LOW'
            })
        })
        
        # Map base types to their subtypes
        self.base_to_subtypes = defaultdict(set)  # {"190R": {"190r-1", "190r-2"}}
        
        self.total_rows = 0
        self.total_true_validations = 0
    
    def learn(self):
        """Main learning function"""
        logger.info("="*60)
        logger.info("STARTING PATTERN LEARNING")
        logger.info("="*60)
        
        # Learn from Excel TRUE results first (most reliable)
        logger.info("Step 1: Learning from Excel TRUE validations...")
        self.learn_from_excel()
        
        # Learn from CSV logs to fill gaps
        logger.info("Step 2: Learning from CSV raw logs...")
        self.learn_from_csv()
        
        # Analyze and save
        logger.info("Step 3: Analyzing patterns and saving...")
        self.analyze_and_save()
        
        logger.info(f"Learning complete! Saved to {self.definitions_file}")
        return self.definitions_file
    
    def learn_from_excel(self):
        """Learn from Excel TRUE results"""
        excel_files = list(self.excel_folder.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files")
        
        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file)
                
                # Find message type columns [XXr] format
                message_cols = [col for col in df.columns if re.match(r'\[.*\]', str(col))]
                
                for _, row in df.iterrows():
                    unit_id = str(row.get('unit_id', ''))
                    station = str(row.get('station', ''))
                    save = str(row.get('save', ''))
                    
                    for msg_col in message_cols:
                        cell_value = str(row[msg_col])
                        
                        # Process TRUE results to learn valid patterns
                        if 'TRUE' in cell_value and 'MIXED' not in cell_value and 'FALSE' not in cell_value:
                            # Get message type from Excel column (e.g., "190r-2")
                            excel_msg_type = msg_col.strip('[]').upper()  # Standardize to uppercase
                            
                            # Update count
                            self.message_types[excel_msg_type]['excel_type'] = excel_msg_type
                            self.message_types[excel_msg_type]['true_count'] += 1
                            self.total_true_validations += 1
                            
                            # Extract timestamps if present
                            timestamp_pattern = r'at timestamps \[([\d.,\s]+)\]'
                            match = re.search(timestamp_pattern, cell_value)
                            if match:
                                timestamps = [float(t.strip()) for t in match.group(1).split(',') if t.strip()]
                                self.message_types[excel_msg_type]['true_timestamps'].extend(timestamps)
                            
                            # Look up actual values from CSV
                            self.lookup_true_values_from_csv(
                                excel_msg_type, unit_id, station, save
                            )
                            
            except Exception as e:
                logger.error(f"Error processing {excel_file.name}: {e}")
        
        logger.info(f"Learned from {self.total_true_validations} TRUE validations")
    
    def lookup_true_values_from_csv(self, excel_msg_type: str, unit_id: str, station: str, save: str):
        """Look up actual values from CSV for a TRUE validation"""
        
        # Build CSV filename
        bus_type = 'rt' if station.startswith('R') else 'lt'
        station_num = re.search(r'\d+', station)
        suffix = (station_num.group() if station_num else '01').zfill(2)
        
        csv_file = self.csv_folder / f"{unit_id}_{station}_{save}_{bus_type}{suffix}.csv"
        
        if not csv_file.exists():
            logger.debug(f"CSV not found: {csv_file.name}")
            return
        
        # Track this as a TRUE source
        self.message_types[excel_msg_type]['true_sources'].add(csv_file.name)
        
        try:
            # Read CSV to find examples
            df = pd.read_csv(csv_file)
            
            # Use timestamps if we have them, otherwise sample
            if self.message_types[excel_msg_type]['true_timestamps']:
                # Look up specific timestamps
                for timestamp in self.message_types[excel_msg_type]['true_timestamps'][:10]:  # Limit to 10
                    time_mask = (df['timestamp'] >= timestamp - 0.001) & \
                               (df['timestamp'] <= timestamp + 0.001)
                    matching_rows = df[time_mask]
                    
                    for _, row in matching_rows.iterrows():
                        self.process_csv_row_for_type(excel_msg_type, row, is_true=True)
            else:
                # Sample the file to find examples
                sample_df = df.sample(min(100, len(df)))
                for _, row in sample_df.iterrows():
                    # We need to figure out if this row matches our excel_msg_type
                    # This is tricky without timestamps, so we'll be conservative
                    self.process_csv_row_for_type(excel_msg_type, row, is_true=False)
                    
        except Exception as e:
            logger.debug(f"Error reading {csv_file.name}: {e}")
    
    def process_csv_row_for_type(self, excel_msg_type: str, row: pd.Series, is_true: bool = False):
        """Process a CSV row to learn patterns for a message type"""
        
        # Extract base type from decoded description
        if 'decoded description' in row and pd.notna(row['decoded description']):
            desc = str(row['decoded description'])
            match = re.search(r'\[([^\]]+)\]', desc)
            if match:
                base_type = match.group(1).upper()  # e.g., "190R"
                
                # Store base type
                if not self.message_types[excel_msg_type]['base_type']:
                    self.message_types[excel_msg_type]['base_type'] = base_type
                
                # Map base type to Excel type
                self.base_to_subtypes[base_type].add(excel_msg_type)
        
        # Get message name (human readable)
        if 'message_name' in row and pd.notna(row['message_name']):
            message_name = str(row['message_name'])
            
            # Check if message_name == decoded_description (indicates data01 error)
            if 'decoded description' in row:
                if message_name != str(row['decoded description']):
                    # Valid message name
                    self.message_types[excel_msg_type]['message_name'] = message_name
        
        # Get data01 value (subtype identifier)
        if 'data01' in row and pd.notna(row['data01']):
            data01_value = str(row['data01'])
            if not self.message_types[excel_msg_type]['data01_value']:
                self.message_types[excel_msg_type]['data01_value'] = data01_value
        
        # Learn all data word values
        for col in row.index:
            if col.startswith('data') and pd.notna(row[col]):
                value = str(row[col])
                self.message_types[excel_msg_type]['data_words'][col]['values'].add(value)
                self.message_types[excel_msg_type]['data_words'][col]['count'] += 1
                
                if is_true:
                    # This is from a TRUE validation - high confidence
                    self.message_types[excel_msg_type]['data_words'][col]['true_values'].add(value)
                    self.message_types[excel_msg_type]['data_words'][col]['confidence'] = 'HIGH'
    
    def learn_from_csv(self):
        """Learn additional patterns from CSV files"""
        csv_files = list(self.csv_folder.glob("*.csv"))[:20]  # Sample first 20 files
        logger.info(f"Sampling {len(csv_files)} CSV files for additional patterns")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=SAMPLE_SIZE)
                
                if 'decoded description' not in df.columns:
                    continue
                
                for _, row in df.iterrows():
                    self.total_rows += 1
                    
                    # Get base type from decoded description
                    desc = str(row.get('decoded description', ''))
                    match = re.search(r'\[([^\]]+)\]', desc)
                    if not match:
                        continue
                    
                    base_type = match.group(1).upper()  # e.g., "190R"
                    
                    # Get data01 to determine subtype
                    data01_value = str(row.get('data01', '')) if 'data01' in row and pd.notna(row['data01']) else None
                    
                    # Try to match to an Excel type based on base_type and data01
                    excel_type = self.find_excel_type(base_type, data01_value)
                    
                    if excel_type:
                        # Update count
                        self.message_types[excel_type]['count'] += 1
                        self.message_types[excel_type]['files_seen'].add(csv_file.name)
                        
                        # Process this row for the identified type
                        self.process_csv_row_for_type(excel_type, row, is_true=False)
                    else:
                        # Unknown subtype - might be a type not in Excel yet
                        # Create a generic entry
                        generic_type = base_type
                        if generic_type not in self.message_types:
                            self.message_types[generic_type]['base_type'] = base_type
                            self.message_types[generic_type]['excel_type'] = generic_type
                            self.message_types[generic_type]['data01_value'] = data01_value
                        
                        self.process_csv_row_for_type(generic_type, row, is_true=False)
                
            except Exception as e:
                logger.debug(f"Error processing {csv_file.name}: {e}")
        
        logger.info(f"Processed {self.total_rows:,} rows from CSV files")
    
    def find_excel_type(self, base_type: str, data01_value: Optional[str]) -> Optional[str]:
        """Find the Excel type that matches this base type and data01 value"""
        
        # Get all Excel types for this base type
        possible_types = self.base_to_subtypes.get(base_type, set())
        
        if not possible_types:
            # No subtypes known, might be a simple type like "01R"
            if base_type in self.message_types:
                return base_type
            return None
        
        # Match by data01 value
        for excel_type in possible_types:
            if self.message_types[excel_type]['data01_value'] == data01_value:
                return excel_type
        
        # No match found
        return None
    
    def analyze_and_save(self):
        """Analyze patterns and save definitions"""
        
        # Analyze patterns for each message type
        for msg_type, info in self.message_types.items():
            # Determine confidence for each data word
            for dw, dw_info in info['data_words'].items():
                if dw_info['true_values']:
                    dw_info['confidence'] = 'HIGH'
                elif dw_info['count'] > 100:
                    dw_info['confidence'] = 'MEDIUM'
                else:
                    dw_info['confidence'] = 'LOW'
        
        # Create output structure
        definitions = {}
        
        for msg_type, info in self.message_types.items():
            # Create definition entry
            definition = {
                'excel_type': msg_type,  # e.g., "190R-2" 
                'base_type': info['base_type'],  # e.g., "190R"
                'data01_identifier': info['data01_value'],  # e.g., "0x02"
                'message_name': info['message_name'],  # e.g., "STORE"
                'occurrences': info['count'],
                'true_validations': info['true_count'],
                'files_seen': len(info['files_seen']),
                'true_source_files': list(info['true_sources'])[:5],  # Sample of TRUE sources
                'data_words': {}
            }
            
            # Process data words
            for dw, dw_info in info['data_words'].items():
                # Use true values if available, otherwise all values
                values = list(dw_info['true_values'] if dw_info['true_values'] else dw_info['values'])
                
                if len(values) == 0:
                    continue
                elif len(values) == 1:
                    definition['data_words'][dw] = {
                        'type': 'single_value',
                        'value': values[0],
                        'confidence': dw_info['confidence']
                    }
                elif len(values) <= 50:
                    definition['data_words'][dw] = {
                        'type': 'multiple_values',
                        'values': sorted(values)[:50],  # Limit to 50
                        'confidence': dw_info['confidence']
                    }
                else:
                    # Check if it's a sequence
                    if self.is_sequence(values):
                        seq_info = self.get_sequence_info(values)
                        definition['data_words'][dw] = {
                            'type': 'sequence',
                            'min': seq_info['min'],
                            'max': seq_info['max'],
                            'step': seq_info.get('step', 1),
                            'confidence': dw_info['confidence']
                        }
                    else:
                        definition['data_words'][dw] = {
                            'type': 'dynamic',
                            'sample_values': sorted(values)[:20],
                            'total_unique': len(values),
                            'confidence': dw_info['confidence']
                        }
            
            definitions[msg_type] = definition
        
        # Create final output
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_rows_processed': self.total_rows,
                'total_true_validations': self.total_true_validations,
                'unique_message_types': len(definitions)
            },
            'base_type_mapping': {k: list(v) for k, v in self.base_to_subtypes.items()},
            'message_definitions': definitions
        }
        
        # Save to file
        with open(self.definitions_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Saved {len(definitions)} message type definitions")
        
        # Print summary
        logger.info("\nSummary of learned message types:")
        high_confidence = sum(1 for d in definitions.values() if d['true_validations'] > 0)
        logger.info(f"  - High confidence (from TRUE): {high_confidence}")
        logger.info(f"  - Total types: {len(definitions)}")
        
        # Show sample of types with subtypes
        for base_type, subtypes in list(self.base_to_subtypes.items())[:5]:
            if len(subtypes) > 1:
                logger.info(f"  - {base_type} has subtypes: {', '.join(sorted(subtypes))}")
    
    def is_sequence(self, values: List[str]) -> bool:
        """Check if values form a sequence"""
        try:
            nums = sorted([float(v) for v in values])
            if len(nums) < 3:
                return False
            
            # Check if differences are consistent
            diffs = [nums[i+1] - nums[i] for i in range(min(10, len(nums)-1))]
            unique_diffs = set(round(d, 3) for d in diffs)
            
            return len(unique_diffs) <= 2  # Allow for wraparound
            
        except (ValueError, TypeError):
            return False
    
    def get_sequence_info(self, values: List[str]) -> Dict:
        """Get sequence information"""
        try:
            nums = [float(v) for v in values]
            return {
                'min': min(nums),
                'max': max(nums),
                'step': 1  # Could enhance to detect actual step
            }
        except:
            return {'min': 0, 'max': 100, 'step': 1}


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
        self.base_type_mapping = data.get('base_type_mapping', {})
        self.metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(self.definitions)} message definitions")
        
        # Create lookup by base type and data01 for fast validation
        self.type_lookup = {}
        for excel_type, definition in self.definitions.items():
            base_type = definition.get('base_type')
            data01_val = definition.get('data01_identifier')
            if base_type and data01_val:
                key = f"{base_type}_{data01_val}"
                self.type_lookup[key] = excel_type
    
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
            
            # Find message type columns
            message_cols = [col for col in df.columns if re.match(r'\[.*\]', str(col))]
            
            for _, row in df.iterrows():
                for msg_col in message_cols:
                    cell_value = str(row[msg_col])
                    
                    # Skip TRUE results
                    if 'TRUE' in cell_value and 'MIXED' not in cell_value and 'FALSE' not in cell_value:
                        continue
                    
                    # Process MIXED or FALSE
                    if 'MIXED' in cell_value or 'FALSE' in cell_value:
                        # Get message type from Excel column
                        excel_msg_type = msg_col.strip('[]').upper()  # Standardize to uppercase
                        
                        # Parse errors
                        errors = self.parse_errors(
                            cell_value, excel_msg_type,
                            str(row.get('unit_id', '')),
                            str(row.get('station', '')),
                            str(row.get('save', ''))
                        )
                        
                        # Validate each error
                        for error in errors:
                            result = self.validate_error(error)
                            self.results.append(result)
                            
        except Exception as e:
            logger.error(f"Error processing {excel_file.name}: {e}")
    
    def parse_errors(self, cell_text: str, excel_msg_type: str,
                    unit_id: str, station: str, save: str) -> List[MessageError]:
        """Parse errors from cell text"""
        errors = []
        
        # Find all data word errors
        pattern = r'messages had incorrect format for (data\d+)\.\s*Expected any of \[(.*?)\].*?found \[(.*?)\]\.\s*Fails occurred at timestamps \[([\d.,\s]+)\]'
        
        for match in re.finditer(pattern, cell_text, re.DOTALL):
            data_word = match.group(1)
            expected = [v.strip().strip("'\"") for v in match.group(2).split(',') if v.strip()]
            found = [v.strip().strip("'\"") for v in match.group(3).split(',') if v.strip()]
            timestamps = [float(t.strip()) for t in match.group(4).split(',') if t.strip()]
            
            for timestamp in timestamps:
                errors.append(MessageError(
                    message_type=excel_msg_type,
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
        
        # Get message definition
        msg_def = self.definitions.get(error.message_type, {})
        base_type = msg_def.get('base_type', error.message_type)
        data01_identifier = msg_def.get('data01_identifier')
        
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
            
            # Find messages in time window
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
            
            # Find error message - look for base type in decoded description
            error_bus = None
            error_row = None
            
            for _, row in window.iterrows():
                # Check decoded description for base type
                if 'decoded description' in row:
                    desc = str(row['decoded description'])
                    if base_type and base_type in desc.upper():
                        # Check if data01 matches (for subtypes)
                        if data01_identifier and 'data01' in row:
                            if str(row['data01']) != data01_identifier:
                                continue
                        
                        # Check if this has the error
                        if abs(row['timestamp'] - error.timestamp) < 0.001:
                            if error.data_word in df.columns:
                                if str(row[error.data_word]) in error.found_values:
                                    error_bus = row['bus']
                                    error_row = row
                                    break
            
            if not error_bus:
                return ValidationResult(
                    error=error,
                    status="ERROR_NOT_FOUND",
                    error_bus=None,
                    opposite_bus=None,
                    correction_timestamp=None,
                    details="Could not find error message"
                )
            
            # Check opposite bus for correction
            opposite_bus = 'B' if error_bus == 'A' else 'A'
            
            # Look for correction on opposite bus
            correction_found = False
            correction_time = None
            
            opposite_msgs = window[window['bus'] == opposite_bus]
            for _, row in opposite_msgs.iterrows():
                # Check if same message type
                if 'decoded description' in row:
                    desc = str(row['decoded description'])
                    if base_type and base_type in desc.upper():
                        # Check data01 for subtype match
                        if data01_identifier and 'data01' in row:
                            if str(row['data01']) != data01_identifier:
                                continue
                        
                        # Check if error is corrected
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
                    confidence='HIGH' if msg_def.get('true_validations', 0) > 0 else 'LOW',
                    details=f"Corrected on Bus {opposite_bus}"
                )
            else:
                return ValidationResult(
                    error=error,
                    status="TRUE_FAILURE",
                    error_bus=error_bus,
                    opposite_bus=None,
                    correction_timestamp=None,
                    confidence='HIGH' if msg_def.get('true_validations', 0) > 0 else 'LOW',
                    details="No correction found on opposite bus"
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
                'Confidence': r.confidence,
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
        
        # Show confidence breakdown
        high_conf = sum(1 for r in self.results if r.confidence == 'HIGH')
        low_conf = sum(1 for r in self.results if r.confidence == 'LOW')
        logger.info(f"High confidence validations: {high_conf}")
        logger.info(f"Low confidence validations: {low_conf}")


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
