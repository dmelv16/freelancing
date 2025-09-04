#!/usr/bin/env python3
"""
Enhanced Message Pattern Learning Script
Learns from both CSV logs AND Excel TRUE results for comprehensive pattern discovery
"""

import pandas as pd
import numpy as np
import re
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMessagePatternLearner:
    """Learns message patterns from both CSV logs and Excel TRUE results"""
    
    def __init__(self, csv_folder: str, excel_folder: str, output_file: str = "message_definitions.json"):
        self.csv_folder = Path(csv_folder)
        self.excel_folder = Path(excel_folder)
        self.output_file = output_file
        
        # Data structures for learning
        self.message_types = defaultdict(lambda: {
            'count': 0,
            'true_count': 0,  # Count of TRUE validations
            'files_seen': set(),
            'data_words': defaultdict(lambda: {
                'values': set(),
                'count': 0,
                'true_values': set(),  # Values seen in TRUE cases
                'is_sequence': False,
                'sequence_info': {},
                'is_static': True,
                'value_distribution': defaultdict(int),
                'confidence': 'LOW'  # Confidence level
            }),
            'true_timestamps': [],  # Timestamps where validation was TRUE
            'data01_identifier': None,
            'base_type': None
        })
        
        # Track relationships
        self.base_type_mapping = defaultdict(set)
        
        # Statistics
        self.total_csv_files = 0
        self.total_excel_files = 0
        self.total_rows_processed = 0
        self.total_true_validations = 0
        
    def learn_from_all_sources(self, sample_size: Optional[int] = None):
        """Learn from both Excel TRUE results and CSV logs"""
        
        # Step 1: Learn from Excel TRUE results first (most reliable)
        logger.info("=" * 60)
        logger.info("STEP 1: Learning from Excel TRUE results")
        logger.info("=" * 60)
        self.learn_from_excel_files()
        
        # Step 2: Learn from CSV logs to fill gaps
        logger.info("=" * 60)
        logger.info("STEP 2: Learning from CSV raw logs")
        logger.info("=" * 60)
        self.learn_from_csv_files(sample_size)
        
        # Step 3: Analyze and combine patterns
        logger.info("=" * 60)
        logger.info("STEP 3: Analyzing combined patterns")
        logger.info("=" * 60)
        self.analyze_patterns()
        
    def learn_from_excel_files(self):
        """Learn from Excel files - extract TRUE validations"""
        excel_files = list(self.excel_folder.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files to analyze")
        
        for idx, excel_file in enumerate(excel_files, 1):
            logger.info(f"Processing Excel {idx}/{len(excel_files)}: {excel_file.name}")
            try:
                self.process_excel_file(excel_file)
                self.total_excel_files += 1
            except Exception as e:
                logger.error(f"Error processing {excel_file.name}: {e}")
        
        logger.info(f"Learned from {self.total_true_validations:,} TRUE validations")
    
    def process_excel_file(self, excel_path: Path):
        """Process a single Excel file for TRUE results"""
        df = pd.read_excel(excel_path)
        
        # Find message type columns
        message_cols = [col for col in df.columns if re.match(r'\[.*\]', str(col))]
        
        for idx, row in df.iterrows():
            unit_id = str(row.get('unit_id', ''))
            station = str(row.get('station', ''))
            save = str(row.get('save', ''))
            
            for msg_col in message_cols:
                cell_value = str(row[msg_col])
                
                # Look for TRUE results
                if 'TRUE' in cell_value and 'MIXED' not in cell_value and 'FALSE' not in cell_value:
                    # Extract message type
                    message_type = msg_col.strip('[]')
                    
                    # Parse TRUE validation info
                    self.learn_from_true_result(
                        message_type, cell_value, 
                        unit_id, station, save
                    )
    
    def learn_from_true_result(self, message_type: str, cell_text: str,
                              unit_id: str, station: str, save: str):
        """Learn from a TRUE validation result"""
        
        # Update message type info
        self.message_types[message_type]['true_count'] += 1
        self.total_true_validations += 1
        
        # Extract timestamps if mentioned
        timestamp_pattern = r'at timestamps \[([\d.,\s]+)\]'
        match = re.search(timestamp_pattern, cell_text)
        if match:
            timestamps = [float(t.strip()) for t in match.group(1).split(',') if t.strip()]
            self.message_types[message_type]['true_timestamps'].extend(timestamps)
            
            # Now we need to look up the actual data values at these timestamps
            self.lookup_true_data_values(
                message_type, timestamps, unit_id, station, save
            )
    
    def lookup_true_data_values(self, message_type: str, timestamps: List[float],
                               unit_id: str, station: str, save: str):
        """Look up actual data values from CSV for TRUE validations"""
        
        # Construct CSV filename
        bus_type = 'rt' if station.startswith('R') else 'lt'
        station_num = re.search(r'\d+', station)
        station_suffix = station_num.group() if station_num else '01'
        station_suffix = station_suffix.zfill(2)
        
        csv_filename = f"{unit_id}_{station}_{save}_{bus_type}{station_suffix}.csv"
        csv_path = self.csv_folder / csv_filename
        
        if not csv_path.exists():
            logger.debug(f"CSV not found for TRUE result: {csv_filename}")
            return
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Look up each timestamp
            for timestamp in timestamps:
                # Find rows near this timestamp
                time_mask = (df['timestamp'] >= timestamp - 0.001) & \
                           (df['timestamp'] <= timestamp + 0.001)
                
                matching_rows = df[time_mask]
                
                for _, row in matching_rows.iterrows():
                    # Check if this row matches our message type
                    if 'message_name' in row and pd.notna(row['message_name']):
                        if str(row['message_name']) == message_type:
                            # This is a TRUE example! Learn all data words
                            self.learn_data_words_from_row(message_type, row, is_true=True)
                            
                            # Track data01 identifier for subtypes
                            if 'data01' in row and pd.notna(row['data01']):
                                self.message_types[message_type]['data01_identifier'] = str(row['data01'])
                            
                            # Track base type
                            if 'decoded description' in row:
                                desc = str(row['decoded description'])
                                match = re.search(r'\[([^\]]+)\]', desc)
                                if match:
                                    base_type = match.group(1)
                                    self.message_types[message_type]['base_type'] = base_type
                                    self.base_type_mapping[base_type].add(message_type)
                    
        except Exception as e:
            logger.debug(f"Error reading CSV for TRUE lookup: {e}")
    
    def learn_data_words_from_row(self, message_type: str, row: pd.Series, is_true: bool = False):
        """Learn all data word values from a row"""
        
        for col in row.index:
            if col.startswith('data') and pd.notna(row[col]):
                value = str(row[col])
                
                # Update data word info
                msg_data = self.message_types[message_type]['data_words'][col]
                msg_data['values'].add(value)
                msg_data['count'] += 1
                msg_data['value_distribution'][value] += 1
                
                if is_true:
                    # This value was seen in a TRUE validation - high confidence!
                    msg_data['true_values'].add(value)
    
    def learn_from_csv_files(self, sample_size: Optional[int] = None):
        """Learn patterns from CSV files"""
        csv_files = list(self.csv_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        # Sample files if we have too many
        if len(csv_files) > 50:
            import random
            csv_files = random.sample(csv_files, 50)
            logger.info(f"Sampling 50 CSV files for learning")
        
        for idx, csv_file in enumerate(csv_files, 1):
            logger.debug(f"Processing CSV {idx}/{len(csv_files)}: {csv_file.name}")
            try:
                self.process_csv_file(csv_file, sample_size)
                self.total_csv_files += 1
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
        
        logger.info(f"Processed {self.total_csv_files} CSV files, {self.total_rows_processed:,} rows")
    
    def process_csv_file(self, csv_path: Path, sample_size: Optional[int] = None):
        """Process a single CSV file"""
        
        # Read file
        if sample_size:
            df = pd.read_csv(csv_path, nrows=sample_size)
        else:
            df = pd.read_csv(csv_path)
        
        if 'message_name' not in df.columns:
            logger.warning(f"No message_name column in {csv_path.name}")
            return
        
        # Track sequences
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        prev_values = {}
        
        for idx, row in df.iterrows():
            self.total_rows_processed += 1
            
            # Get message type from message_name
            if pd.notna(row['message_name']):
                message_type = str(row['message_name'])
                
                # Skip if message_name fell back to decoded_description
                if 'decoded description' in row and message_type == str(row['decoded description']):
                    continue  # This is an error case
                
                # Update count
                self.message_types[message_type]['count'] += 1
                self.message_types[message_type]['files_seen'].add(csv_path.name)
                
                # Learn data words
                self.learn_data_words_from_row(message_type, row, is_true=False)
                
                # Check for sequences
                for col in df.columns:
                    if col.startswith('data') and pd.notna(row[col]):
                        value = str(row[col])
                        prev_key = f"{message_type}_{col}"
                        
                        if prev_key in prev_values:
                            self._check_sequence(message_type, col, prev_values[prev_key], value)
                        prev_values[prev_key] = value
    
    def _check_sequence(self, msg_type: str, data_word: str, prev_value: str, curr_value: str):
        """Check if values indicate a sequence"""
        try:
            prev_num = float(prev_value)
            curr_num = float(curr_value)
            
            diff = curr_num - prev_num
            
            # Track differences
            if 'diffs' not in self.message_types[msg_type]['data_words'][data_word]:
                self.message_types[msg_type]['data_words'][data_word]['diffs'] = []
            
            self.message_types[msg_type]['data_words'][data_word]['diffs'].append(diff)
            
        except (ValueError, TypeError):
            pass
    
    def analyze_patterns(self):
        """Analyze collected patterns and determine confidence levels"""
        
        for msg_type, msg_info in self.message_types.items():
            logger.debug(f"Analyzing {msg_type}: {msg_info['count']} occurrences, {msg_info['true_count']} TRUE")
            
            for data_word, word_info in msg_info['data_words'].items():
                values = word_info['values']
                true_values = word_info['true_values']
                
                # Determine confidence level
                if true_values:
                    # We have TRUE validation examples - high confidence
                    word_info['confidence'] = 'HIGH'
                elif word_info['count'] > 100:
                    # Many examples seen
                    word_info['confidence'] = 'MEDIUM'
                else:
                    word_info['confidence'] = 'LOW'
                
                # Determine pattern type
                if len(values) == 1:
                    word_info['pattern_type'] = 'single_value'
                    word_info['is_static'] = True
                elif len(values) <= 20:
                    word_info['pattern_type'] = 'multiple_values'
                    word_info['is_static'] = True
                else:
                    # Check for sequence
                    if self._is_sequence(list(values)):
                        word_info['is_sequence'] = True
                        word_info['pattern_type'] = 'sequence'
                        word_info['sequence_info'] = self._get_sequence_info(list(values))
                    else:
                        word_info['pattern_type'] = 'dynamic'
                        word_info['is_static'] = False
    
    def _is_sequence(self, values: List[str]) -> bool:
        """Check if values form a sequence"""
        try:
            nums = sorted([float(v) for v in values])
            
            if len(nums) < 3:
                return False
            
            # Check if evenly spaced
            diffs = [nums[i+1] - nums[i] for i in range(min(10, len(nums)-1))]
            unique_diffs = set(round(d, 3) for d in diffs)
            
            return len(unique_diffs) <= 2  # Allow for wraparound
            
        except (ValueError, TypeError):
            return False
    
    def _get_sequence_info(self, values: List[str]) -> Dict:
        """Get sequence information"""
        try:
            nums = [float(v) for v in values]
            return {
                'min': min(nums),
                'max': max(nums),
                'unique_count': len(set(nums)),
                'step': self._detect_step(sorted(nums))
            }
        except (ValueError, TypeError):
            return {}
    
    def _detect_step(self, sorted_nums: List[float]) -> float:
        """Detect step size in sequence"""
        if len(sorted_nums) < 2:
            return 1
        
        diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(min(10, len(sorted_nums)-1))]
        
        # Return most common difference
        from collections import Counter
        counter = Counter([round(d, 3) for d in diffs])
        most_common, _ = counter.most_common(1)[0]
        return most_common
    
    def generate_definition_file(self):
        """Generate the comprehensive message definition file"""
        logger.info(f"Generating definition file: {self.output_file}")
        
        definitions = {}
        
        for msg_type, msg_info in self.message_types.items():
            definition = {
                'message_type': msg_type,
                'base_type': msg_info.get('base_type', ''),
                'occurrences': msg_info['count'],
                'true_validations': msg_info['true_count'],
                'files_seen': len(msg_info['files_seen']),
                'true_source_files': list(msg_info.get('true_sources', [])),  # CSV files with TRUE examples
                'true_examples': msg_info.get('true_examples', [])[:10],  # Sample of (unit_id, station, save)
                'data_words': {}
            }
            
            # Add data01 identifier if present
            if msg_info.get('data01_identifier'):
                definition['subtype_identifier'] = {
                    'column': 'data01',
                    'value': msg_info['data01_identifier']
                }
            
            # Process each data word
            for data_word, word_info in msg_info['data_words'].items():
                all_values = list(word_info['values'])
                true_values = list(word_info['true_values'])
                
                dw_def = {
                    'confidence': word_info['confidence'],
                    'occurrences': word_info['count'],
                    'pattern_type': word_info.get('pattern_type', 'unknown')
                }
                
                # Use TRUE values if available, otherwise all values
                values_to_use = true_values if true_values else all_values
                
                if word_info.get('is_sequence'):
                    dw_def['type'] = 'sequence'
                    dw_def['min'] = word_info['sequence_info'].get('min', 0)
                    dw_def['max'] = word_info['sequence_info'].get('max', 100)
                    dw_def['step'] = word_info['sequence_info'].get('step', 1)
                elif len(values_to_use) == 1:
                    dw_def['type'] = 'single_value'
                    dw_def['value'] = values_to_use[0]
                elif len(values_to_use) <= 50:
                    dw_def['type'] = 'multiple_values'
                    dw_def['values'] = sorted(values_to_use)
                    if true_values:
                        dw_def['true_values'] = sorted(true_values)
                else:
                    dw_def['type'] = 'dynamic'
                    dw_def['sample_values'] = sorted(values_to_use)[:20]
                    dw_def['total_unique'] = len(values_to_use)
                    if true_values:
                        dw_def['true_values'] = sorted(true_values)[:20]
                
                definition['data_words'][data_word] = dw_def
            
            definitions[msg_type] = definition
        
        # Create output structure
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_csv_files': self.total_csv_files,
                'total_excel_files': self.total_excel_files,
                'total_rows': self.total_rows_processed,
                'total_true_validations': self.total_true_validations,
                'unique_message_types': len(definitions)
            },
            'message_definitions': definitions,
            'base_type_mapping': {k: list(v) for k, v in self.base_type_mapping.items()}
        }
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Saved {len(definitions)} message type definitions")
        
        # Generate summary report
        self.generate_summary_report(definitions)
    
    def generate_summary_report(self, definitions: Dict):
        """Generate human-readable summary report"""
        report_file = self.output_file.replace('.json', '_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED MESSAGE PATTERN LEARNING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Excel Files Processed: {self.total_excel_files}\n")
            f.write(f"Total CSV Files Processed: {self.total_csv_files}\n")
            f.write(f"Total TRUE Validations Learned From: {self.total_true_validations:,}\n")
            f.write(f"Total Rows Processed: {self.total_rows_processed:,}\n")
            f.write(f"Unique Message Types Found: {len(definitions)}\n\n")
            
            # Confidence breakdown
            high_conf = sum(1 for d in definitions.values() 
                          for dw in d['data_words'].values() 
                          if dw.get('confidence') == 'HIGH')
            med_conf = sum(1 for d in definitions.values() 
                         for dw in d['data_words'].values() 
                         if dw.get('confidence') == 'MEDIUM')
            low_conf = sum(1 for d in definitions.values() 
                         for dw in d['data_words'].values() 
                         if dw.get('confidence') == 'LOW')
            
            f.write("CONFIDENCE LEVELS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"HIGH confidence data words: {high_conf} (from TRUE validations)\n")
            f.write(f"MEDIUM confidence data words: {med_conf}\n")
            f.write(f"LOW confidence data words: {low_conf}\n\n")
            
            # Messages with most TRUE validations
            f.write("TOP 10 MOST VALIDATED MESSAGE TYPES:\n")
            f.write("-" * 40 + "\n")
            sorted_by_true = sorted(definitions.items(), 
                                  key=lambda x: x[1].get('true_validations', 0), 
                                  reverse=True)[:10]
            
            for msg_type, definition in sorted_by_true:
                true_count = definition.get('true_validations', 0)
                total_count = definition.get('occurrences', 0)
                if true_count > 0:
                    f.write(f"{msg_type}: {true_count:,} TRUE validations, {total_count:,} total\n")
            
            f.write("\n")
            
            # Pattern type statistics
            sequence_count = 0
            static_count = 0
            dynamic_count = 0
            
            for definition in definitions.values():
                for dw_def in definition.get('data_words', {}).values():
                    pattern = dw_def.get('pattern_type', '')
                    if pattern == 'sequence':
                        sequence_count += 1
                    elif pattern in ['single_value', 'multiple_values']:
                        static_count += 1
                    elif pattern == 'dynamic':
                        dynamic_count += 1
            
            f.write("PATTERN TYPE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Static data words: {static_count}\n")
            f.write(f"Sequential data words: {sequence_count}\n")
            f.write(f"Dynamic data words: {dynamic_count}\n")
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Learn message patterns from Excel TRUE results and CSV logs'
    )
    parser.add_argument(
        '--csv-folder',
        required=True,
        help='Folder containing CSV raw logs'
    )
    parser.add_argument(
        '--excel-folder',
        required=True,
        help='Folder containing Excel validation reports'
    )
    parser.add_argument(
        '--output',
        default='enhanced_message_definitions.json',
        help='Output file for message definitions'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of rows to sample per CSV file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create learner
    learner = EnhancedMessagePatternLearner(
        csv_folder=args.csv_folder,
        excel_folder=args.excel_folder,
        output_file=args.output
    )
    
    # Learn from all sources
    logger.info("Starting enhanced pattern learning...")
    learner.learn_from_all_sources(sample_size=args.sample_size)
    
    # Generate output
    learner.generate_definition_file()
    
    logger.info("Enhanced learning complete!")

if __name__ == "__main__":
    main()
