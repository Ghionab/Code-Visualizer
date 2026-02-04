import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.config import Config, VerbosityLevel, OutputFormat
from .core.data_models import *


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='python-visualizer',
        description='Python Code Execution Visualizer - Educational tool for understanding Python internals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python-visualizer --gui                  # Launch graphical interface
  python-visualizer script.py              # Visualize execution of script.py
  python-visualizer -c "print('hello')"   # Visualize inline code
  python-visualizer --verbose script.py   # Detailed explanations
  python-visualizer --no-color script.py  # Plain text output
  python-visualizer --step script.py      # Interactive stepping mode
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        'file',
        nargs='?',
        type=str,
        help='Python file to analyze and visualize'
    )
    input_group.add_argument(
        '-c', '--code',
        type=str,
        help='Python code string to analyze and visualize'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o', '--output',
        type=str,
        help='Output file (default: stdout)'
    )
    output_group.add_argument(
        '--format',
        choices=['console', 'plain', 'json', 'html'],
        default='console',
        help='Output format (default: console)'
    )
    output_group.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    # Verbosity options
    verbosity_group = parser.add_argument_group('Verbosity Options')
    verbosity_group.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity level (use -v, -vv for more detail)'
    )
    verbosity_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimal output mode'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--no-ast',
        action='store_true',
        help='Skip AST analysis and visualization'
    )
    analysis_group.add_argument(
        '--no-bytecode',
        action='store_true',
        help='Skip bytecode analysis'
    )
    analysis_group.add_argument(
        '--no-memory',
        action='store_true',
        help='Skip memory tracking and visualization'
    )
    analysis_group.add_argument(
        '--no-trace',
        action='store_true',
        help='Skip execution tracing'
    )
    
    # Interactive options
    interactive_group = parser.add_argument_group('Interactive Options')
    interactive_group.add_argument(
        '--step',
        action='store_true',
        help='Enable interactive stepping mode'
    )
    interactive_group.add_argument(
        '--auto-step',
        type=float,
        metavar='DELAY',
        help='Automatic stepping with delay in seconds'
    )
    interactive_group.add_argument(
        '--breakpoint',
        type=int,
        action='append',
        metavar='LINE',
        help='Set breakpoint at line number (can be used multiple times)'
    )
    
    # Interface options
    interface_group = parser.add_argument_group('Interface Options')
    interface_group.add_argument(
        '--gui',
        action='store_true',
        help='Launch graphical user interface'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--max-events',
        type=int,
        default=10000,
        help='Maximum number of trace events to capture (default: 10000)'
    )
    advanced_group.add_argument(
        '--max-depth',
        type=int,
        default=100,
        help='Maximum recursion depth to analyze (default: 100)'
    )
    advanced_group.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command-line arguments and check for conflicts."""
    
    # GUI mode doesn't require input validation
    if args.gui:
        return True
    
    # Non-GUI mode requires input
    if not args.file and not args.code:
        print("Error: Must provide either a file or code string (or use --gui for graphical interface).", file=sys.stderr)
        return False
    
    # Check if input file exists
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found.", file=sys.stderr)
            return False
        
        if not file_path.is_file():
            print(f"Error: '{args.file}' is not a regular file.", file=sys.stderr)
            return False
        
        # Check if it's a Python file
        if file_path.suffix not in ['.py', '.pyw']:
            print(f"Warning: '{args.file}' does not appear to be a Python file.")
    
    # Validate output file path
    if args.output:
        output_path = Path(args.output)
        try:
            # Check if we can write to the output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(f"Error: Cannot write to output directory '{output_path.parent}'.", file=sys.stderr)
            return False
    
    # Check for conflicting options
    if args.quiet and args.verbose > 0:
        print("Error: Cannot use --quiet and --verbose together.", file=sys.stderr)
        return False
    
    if args.step and args.auto_step is not None:
        print("Error: Cannot use --step and --auto-step together.", file=sys.stderr)
        return False
    
    # Validate numeric arguments
    if args.auto_step is not None and args.auto_step < 0:
        print("Error: Auto-step delay must be non-negative.", file=sys.stderr)
        return False
    
    if args.max_events < 1:
        print("Error: Maximum events must be at least 1.", file=sys.stderr)
        return False
    
    if args.max_depth < 1:
        print("Error: Maximum depth must be at least 1.", file=sys.stderr)
        return False
    
    return True


def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create a configuration object from parsed command-line arguments."""
    
    config = Config()
    
    # Set input source
    if args.file:
        config.input_file = args.file
        # Read the source code
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                config.source_code = f.read()
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.code:
        config.source_code = args.code
    
    # Set output options
    if args.output:
        config.output_file = args.output
    
    config.output_format = OutputFormat(args.format)
    
    if args.no_color:
        config.display.use_colors = False
    
    # Set verbosity
    if args.quiet:
        config.verbosity = VerbosityLevel.MINIMAL
    elif args.verbose == 1:
        config.verbosity = VerbosityLevel.DETAILED
    elif args.verbose >= 2:
        config.verbosity = VerbosityLevel.EXPERT
    
    # Set analysis options
    if args.no_ast:
        config.analysis.analyze_ast = False
    if args.no_bytecode:
        config.analysis.analyze_bytecode = False
    if args.no_memory:
        config.tracing.trace_memory = False
        config.analysis.analyze_memory_layout = False
    if args.no_trace:
        config.tracing.trace_opcodes = False
        config.tracing.trace_calls = False
        config.tracing.trace_returns = False
    
    # Set interactive options
    if args.step:
        config.interactive.auto_step = False
    elif args.auto_step is not None:
        config.interactive.auto_step = True
        config.interactive.step_delay = args.auto_step
    
    if args.breakpoint:
        config.interactive.breakpoints = set(args.breakpoint)
    
    # Set advanced options
    config.tracing.max_trace_events = args.max_events
    config.analysis.max_recursion_depth = args.max_depth
    
    return config


def main() -> int:
    """Main entry point for the CLI application."""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check if GUI mode is requested
    if args.gui:
        from .gui_launcher import launch_gui
        return launch_gui()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Create configuration
    config = create_config_from_args(args)
    
    # TODO: Initialize and run the visualizer
    # This will be implemented in later tasks
    print("Python Code Execution Visualizer")
    print(f"Configuration created successfully:")
    print(f"  Input: {'file' if config.input_file else 'code string'}")
    print(f"  Output format: {config.output_format.value}")
    print(f"  Verbosity: {config.verbosity.value}")
    print(f"  Colors: {'enabled' if config.display.use_colors else 'disabled'}")
    
    if config.source_code:
        print(f"\nSource code preview (first 100 chars):")
        preview = config.source_code[:100]
        if len(config.source_code) > 100:
            preview += "..."
        print(f"  {repr(preview)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())