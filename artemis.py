#!/usr/bin/env python3
"""
Message Pattern Learning Script
Analyzes raw CSV logs to learn message types and their data word patterns
Outputs a comprehensive message definition file for use in validation
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

class MessagePatternLearner:
    """Learns message patterns from raw CSV logs"""
    
    def __init__(self, csv_folder: str, output_file: str = "message_definitions.json"):
        self.csv_folder = Path(csv_folder)
        self.output_file = output_file
        
        # Data structures for learning
        self.message_types = defaultdict(lambda: {
            'count': 0,
            'files_seen': set(),
            'data_words': defaultdict(lambda: {
                'values': set(),
                'count': 0,
                'is_sequence': False,
                'sequence_info': {},
                'is_static': True,
                'value_distribution': defaultdict(int)
            }),
            'subtypes': defaultdict(lambda: {
                'identifier_column': None,
                'identifier_value': None,
                'count': 0
            })
        })
        
        # Track relationships between base types and hyphenated versions
        self.base_type_mapping = defaultdict(set)
        
        # Statistics
        self.total_files_processed = 0
        self.total_rows_processed = 0
        
    def learn_from_all_files(self, sample_size: Optional[int] = None):
        """
        Process all CSV files to learn patterns
        
        Args:
            sample_size: Number of rows to sample per file (None = all rows)
        """
        csv_files = list(self.csv_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        for idx, csv_file in enumerate(csv_files, 1):
            logger.info(f"Processing file {idx}/{len(csv_files)}: {csv_file.name}")
            try:
                self.learn_from_file(csv_file, sample_size)
                self.total_files_processed += 1
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
        
        logger.info(f"Processed {self.total_files_processed} files, {self.total_rows_processed:,} total rows")
        
        # Analyze patterns after processing all files
        self.analyze_patterns()
        
    def learn_from_file(self, csv_path: Path, sample_size: Optional[int] = None):
        """Learn patterns from a single CSV file"""
        # Read file
        if sample_size:
            df = pd.read_csv(csv_path, nrows=sample_size)
        else:
            df = pd.read_csv(csv_path)
        
        if 'decoded description' not in df.columns:
            logger.warning(f"No 'decoded description' column in {csv_path.name}")
            return
        
        # Group by timestamp to track sequences
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Track previous values for sequence detection
        prev_values = {}
        
        for idx, row in df.iterrows():
            self.total_rows_processed += 1
            
            # Extract message type from decoded description
            desc = str(row.get('decoded description', ''))
            match = re.search(r'\[([^\]]+)\]', desc)
            
            if not match:
                continue
            
            base_type = match.group(1)
            
            # Update message type info
            self.message_types[base_type]['count'] += 1
            self.message_types[base_type]['files_seen'].add(csv_path.name)
            
            # Analyze data words
            for col in df.columns:
                if col.startswith('data') and pd.notna(row[col]):
                    value = str(row[col])
                    
                    # Update data word info
                    msg_data = self.message_types[base_type]['data_words'][col]
                    msg_data['values'].add(value)
                    msg_data['count'] += 1
                    msg_data['value_distribution'][value] += 1
                    
                    # Check for sequences
                    prev_key = f"{base_type}_{col}"
                    if prev_key in prev_values:
                        self._check_sequence(base_type, col, prev_values[prev_key], value)
                    prev_values[prev_key] = value
            
            # Detect hyphenated message types (e.g., 190R-1, 190R-2)
            self._detect_hyphenated_types(base_type, row)
    
    def _check_sequence(self, msg_type: str, data_word: str, prev_value: str, curr_value: str):
        """Check if values indicate a sequence"""
        try:
            prev_num = float(prev_value)
            curr_num = float(curr_value)
            
            diff = curr_num - prev_num
            
            # Track differences to detect patterns
            if 'diffs' not in self.message_types[msg_type]['data_words'][data_word]:
                self.message_types[msg_type]['data_words'][data_word]['diffs'] = []
            
            self.message_types[msg_type]['data_words'][data_word]['diffs'].append(diff)
            
        except (ValueError, TypeError):
            # Not numeric, can't be a sequence
            pass
    
    def _detect_hyphenated_types(self, base_type: str, row: pd.Series):
        """Detect if this is a hyphenated subtype based on data patterns"""
        # Look for patterns that might indicate subtypes
        # For example, if data01 has consistent values that differ between messages
        
        # Check if base type could have hyphens (e.g., "190R" might be "190R-1" or "190R-2")
        base_without_hyphen = base_type.split('-')[0] if '-' in base_type else base_type
        self.base_type_mapping[base_without_hyphen].add(base_type)
        
        # Try to identify subtype based on data01 or other identifying columns
        for col in ['data01', 'data02', 'data03']:
            if col in row and pd.notna(row[col]):
                value = str(row[col])
                subtype_key = f"{base_type}_{value}"
                self.message_types[base_type]['subtypes'][subtype_key]['identifier_column'] = col
                self.message_types[base_type]['subtypes'][subtype_key]['identifier_value'] = value
                self.message_types[base_type]['subtypes'][subtype_key]['count'] += 1
    
    def analyze_patterns(self):
        """Analyze collected data to determine patterns"""
        logger.info("Analyzing patterns...")
        
        for msg_type, msg_info in self.message_types.items():
            logger.debug(f"Analyzing {msg_type}: {msg_info['count']} occurrences")
            
            for data_word, word_info in msg_info['data_words'].items():
                values = word_info['values']
                
                # Determine if static or dynamic
                if len(values) == 1:
                    word_info['is_static'] = True
                    word_info['pattern_type'] = 'single_value'
                elif len(values) <= 10:
                    word_info['is_static'] = True
                    word_info['pattern_type'] = 'multiple_values'
                else:
                    # Check if it's a sequence
                    if self._is_sequence(list(values)):
                        word_info['is_sequence'] = True
                        word_info['pattern_type'] = 'sequence'
                        word_info['sequence_info'] = self._get_sequence_info(list(values))
                    else:
                        word_info['is_static'] = False
                        word_info['pattern_type'] = 'dynamic'
                
                # Analyze differences for sequence detection
                if 'diffs' in word_info and len(word_info['diffs']) > 10:
                    common_diff = self._get_common_difference(word_info['diffs'])
                    if common_diff is not None:
                        word_info['is_sequence'] = True
                        word_info['sequence_step'] = common_diff
    
    def _is_sequence(self, values: List[str]) -> bool:
        """Check if values form a sequence"""
        try:
            # Try to convert to numbers
            nums = sorted([float(v) for v in values])
            
            if len(nums) < 3:
                return False
            
            # Check if evenly spaced
            diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            
            # Allow some tolerance for floating point
            unique_diffs = set(round(d, 3) for d in diffs)
            
            # If all differences are the same (or very close), it's a sequence
            if len(unique_diffs) <= 2:  # Allow for wraparound
                return True
                
        except (ValueError, TypeError):
            pass
        
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
        """Detect the step size in a sequence"""
        if len(sorted_nums) < 2:
            return 1
        
        diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(min(10, len(sorted_nums)-1))]
        # Return the most common difference
        return max(set(diffs), key=diffs.count) if diffs else 1
    
    def _get_common_difference(self, diffs: List[float]) -> Optional[float]:
        """Get the most common difference if it's consistent enough"""
        if not diffs:
            return None
        
        # Round differences to handle floating point
        rounded = [round(d, 3) for d in diffs]
        
        # Find most common
        from collections import Counter
        counter = Counter(rounded)
        most_common, count = counter.most_common(1)[0]
        
        # If >70% of differences are the same, it's likely a sequence
        if count / len(diffs) > 0.7:
            return most_common
        
        return None
    
    def generate_definition_file(self):
        """Generate the message definition file"""
        logger.info(f"Generating definition file: {self.output_file}")
        
        definitions = {}
        
        for msg_type, msg_info in self.message_types.items():
            definition = {
                'base_description': msg_type,
                'occurrences': msg_info['count'],
                'files_seen': len(msg_info['files_seen']),
                'data_words': {}
            }
            
            # Check if this might be a hyphenated type
            if msg_type in self.base_type_mapping and len(self.base_type_mapping[msg_type]) > 1:
                definition['has_subtypes'] = True
                definition['known_subtypes'] = list(self.base_type_mapping[msg_type])
            
            # Process each data word
            for data_word, word_info in msg_info['data_words'].items():
                values = list(word_info['values'])
                
                if word_info.get('is_sequence'):
                    definition['data_words'][data_word] = {
                        'type': 'sequence',
                        'min': word_info['sequence_info'].get('min', 0),
                        'max': word_info['sequence_info'].get('max', 100),
                        'step': word_info['sequence_info'].get('step', 1)
                    }
                elif len(values) == 1:
                    definition['data_words'][data_word] = values[0]  # Single value
                elif len(values) <= 20:  # Reasonable number of options
                    definition['data_words'][data_word] = sorted(values)
                else:
                    # Too many values, might be dynamic
                    definition['data_words'][data_word] = {
                        'type': 'dynamic',
                        'sample_values': sorted(values)[:10],  # Keep sample
                        'total_unique': len(values)
                    }
                
                # Add statistics
                definition['data_words'][f"{data_word}_stats"] = {
                    'occurrences': word_info['count'],
                    'unique_values': len(values),
                    'pattern_type': word_info.get('pattern_type', 'unknown')
                }
            
            definitions[msg_type] = definition
        
        # Handle hyphenated types specially
        self._process_hyphenated_types(definitions)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            json.dump(definitions, f, indent=2, default=str)
        
        logger.info(f"Saved {len(definitions)} message type definitions")
        
        # Also save a summary report
        self.generate_summary_report(definitions)
    
    def _process_hyphenated_types(self, definitions: Dict):
        """Process hyphenated message types (e.g., 190R-1, 190R-2)"""
        # Group by base type
        for base_type, variants in self.base_type_mapping.items():
            if len(variants) > 1:
                logger.info(f"Found variants for {base_type}: {variants}")
                
                # Try to identify what makes them different
                for variant in variants:
                    if variant in definitions:
                        # Look for identifying characteristics
                        # This would need to be customized based on your actual data
                        pass
    
    def generate_summary_report(self, definitions: Dict, subtype_mapping: Dict):
        """Generate a human-readable summary report"""
        report_file = self.output_file.replace('.json', '_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("MESSAGE PATTERN LEARNING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Files Processed: {self.total_files_processed}\n")
            f.write(f"Total Rows Processed: {self.total_rows_processed:,}\n")
            f.write(f"Unique Message Types Found: {len(definitions)}\n\n")
            
            # Report on base types with subtypes
            if subtype_mapping:
                f.write("BASE TYPES WITH SUBTYPES:\n")
                f.write("-" * 40 + "\n")
                for base, info in sorted(subtype_mapping.items()):
                    f.write(f"{base}: {info['count']} subtypes\n")
                    for subtype in sorted(info['subtypes']):
                        if subtype in definitions:
                            identifier = definitions[subtype].get('subtype_identifier', {})
                            if identifier:
                                f.write(f"  - {subtype} (data01='{identifier.get('value', 'unknown')}')\n")
                            else:
                                f.write(f"  - {subtype}\n")
                f.write("\n")
            
            # Report on problematic messages
            problems = [mt for mt, d in definitions.items() if d.get('warning')]
            if problems:
                f.write("MESSAGES WITH DATA01 ISSUES:\n")
                f.write("-" * 40 + "\n")
                for msg_type in sorted(problems):
                    f.write(f"  - {msg_type}\n")
                f.write("\n")
            
            f.write("MESSAGE TYPE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            
            # Statistics
            sequence_count = 0
            static_count = 0
            dynamic_count = 0
            
            for msg_type, definition in sorted(definitions.items()):
                f.write(f"\n{msg_type}:\n")
                f.write(f"  Base description: {definition.get('base_description', 'N/A')}\n")
                f.write(f"  Occurrences: {definition.get('occurrences', 0):,}\n")
                f.write(f"  Files seen in: {definition.get('files_seen', 0)}\n")
                
                if definition.get('subtype_identifier'):
                    ident = definition['subtype_identifier']
                    f.write(f"  Subtype identifier: {ident['column']}='{ident['value']}'\n")
                
                # Analyze data words
                data_words = definition.get('data_words', {})
                
                if data_words:
                    f.write(f"  Data words:\n")
                    
                    for dw, dw_def in sorted(data_words.items()):
                        if isinstance(dw_def, dict):
                            dw_type = dw_def.get('type', 'unknown')
                            if dw_type == 'sequence':
                                f.write(f"    {dw}: SEQUENCE ({dw_def.get('min', 0)}-{dw_def.get('max', 'N')})\n")
                                sequence_count += 1
                            elif dw_type == 'dynamic':
                                f.write(f"    {dw}: DYNAMIC ({dw_def.get('total_unique', 0)} unique values)\n")
                                dynamic_count += 1
                            elif dw_type == 'single_value':
                                f.write(f"    {dw}: STATIC ('{dw_def.get('value', '')}')\n")
                                static_count += 1
                            elif dw_type == 'multiple_values':
                                f.write(f"    {dw}: MULTIPLE ({len(dw_def.get('values', []))} valid options)\n")
                                static_count += 1
            
            f.write(f"\n\nSUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Static data words: {static_count}\n")
            f.write(f"Sequential data words: {sequence_count}\n")
            f.write(f"Dynamic data words: {dynamic_count}\n")
            
            # Find most common message types
            f.write(f"\n\nTOP 10 MOST FREQUENT MESSAGE TYPES:\n")
            f.write("-" * 40 + "\n")
            sorted_by_count = sorted(definitions.items(), 
                                   key=lambda x: x[1].get('occurrences', 0), 
                                   reverse=True)[:10]
            for msg_type, definition in sorted_by_count:
                f.write(f"{msg_type}: {definition.get('occurrences', 0):,} occurrences\n")
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Learn message patterns from raw CSV logs'
    )
    parser.add_argument(
        '--csv-folder',
        required=True,
        help='Folder containing CSV raw logs'
    )
    parser.add_argument(
        '--output',
        default='message_definitions.json',
        help='Output file for message definitions (JSON format)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of rows to sample per file (default: all rows)'
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
    learner = MessagePatternLearner(
        csv_folder=args.csv_folder,
        output_file=args.output
    )
    
    # Learn from all files
    logger.info("Starting pattern learning...")
    learner.learn_from_all_files(sample_size=args.sample_size)
    
    # Generate output
    learner.generate_definition_file()
    
    logger.info("Learning complete!")

if __name__ == "__main__":
    main()
