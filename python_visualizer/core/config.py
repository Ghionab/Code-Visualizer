
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import sys
import os


class VerbosityLevel(Enum):
    """Verbosity levels for educational explanations."""
    MINIMAL = "minimal"
    NORMAL = "normal"
    DETAILED = "detailed"
    EXPERT = "expert"


class OutputFormat(Enum):
    """Output format options."""
    CONSOLE = "console"
    PLAIN_TEXT = "plain"
    JSON = "json"
    HTML = "html"


@dataclass
class DisplayConfig:
    """Configuration for display and formatting options."""
    use_colors: bool = True
    use_unicode: bool = True
    max_line_width: int = 120
    indent_size: int = 2
    show_line_numbers: bool = True
    show_timestamps: bool = False
    
    # Color scheme
    colors: Dict[str, str] = field(default_factory=lambda: {
        'opcode': '\033[94m',      # Blue
        'value': '\033[92m',       # Green
        'memory': '\033[93m',      # Yellow
        'error': '\033[91m',       # Red
        'comment': '\033[90m',     # Gray
        'highlight': '\033[95m',   # Magenta
        'reset': '\033[0m'         # Reset
    })
    
    # ASCII art characters
    ascii_chars: Dict[str, str] = field(default_factory=lambda: {
        'tree_branch': '├── ',
        'tree_last': '└── ',
        'tree_vertical': '│   ',
        'tree_space': '    ',
        'arrow_right': '→ ',
        'arrow_down': '↓ ',
        'bullet': '• '
    })


@dataclass
class TracingConfig:
    """Configuration for execution tracing behavior."""
    trace_opcodes: bool = True
    trace_memory: bool = True
    trace_calls: bool = True
    trace_returns: bool = True
    trace_exceptions: bool = True
    trace_lines: bool = True
    
    # Memory tracking options
    track_allocations: bool = True
    track_reference_counts: bool = True
    track_garbage_collection: bool = True
    
    # Performance limits
    max_trace_events: int = 10000
    max_memory_snapshots: int = 1000


@dataclass
class AnalysisConfig:
    """Configuration for code analysis components."""
    analyze_ast: bool = True
    analyze_bytecode: bool = True
    analyze_control_flow: bool = True
    analyze_memory_layout: bool = True
    analyze_io_operations: bool = True
    
    # Analysis depth limits
    max_recursion_depth: int = 100
    max_object_depth: int = 10


@dataclass
class InteractiveConfig:
    """Configuration for interactive stepping and control."""
    auto_step: bool = False
    step_delay: float = 0.0
    pause_on_exception: bool = True
    pause_on_function_call: bool = False
    pause_on_memory_allocation: bool = False
    
    # Breakpoint settings
    breakpoints: Set[int] = field(default_factory=set)
    break_on_opcodes: Set[str] = field(default_factory=set)


@dataclass
class Config:
    """Main configuration class that combines all settings."""
    
    # Sub-configurations
    display: DisplayConfig = field(default_factory=DisplayConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    interactive: InteractiveConfig = field(default_factory=InteractiveConfig)
    
    # General settings
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    output_format: OutputFormat = OutputFormat.CONSOLE
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    
    # Input/Output settings
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    source_code: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup and validation."""
        self._detect_terminal_capabilities()
        self._validate_settings()
    
    def _detect_terminal_capabilities(self):
        """Detect terminal capabilities and adjust settings accordingly."""
        # Disable colors if not in a TTY or on Windows without ANSI support
        if not sys.stdout.isatty():
            self.display.use_colors = False
        
        # Check for Windows and adjust Unicode support
        if os.name == 'nt':
            try:
                # Test Unicode support
                sys.stdout.write('\u2500')
                sys.stdout.flush()
            except UnicodeEncodeError:
                self.display.use_unicode = False
    
    def _validate_settings(self):
        """Validate configuration settings and apply constraints."""
        # Ensure reasonable limits
        if self.tracing.max_trace_events < 100:
            self.tracing.max_trace_events = 100
        
        if self.display.max_line_width < 40:
            self.display.max_line_width = 40
        
        if self.display.indent_size < 1:
            self.display.indent_size = 1
    
    def update_from_args(self, args: Dict[str, any]):
        """Update configuration from command-line arguments."""
        if 'verbose' in args and args['verbose']:
            if args['verbose'] == 1:
                self.verbosity = VerbosityLevel.DETAILED
            elif args['verbose'] >= 2:
                self.verbosity = VerbosityLevel.EXPERT
        
        if 'no_color' in args and args['no_color']:
            self.display.use_colors = False
        
        if 'output_format' in args:
            try:
                self.output_format = OutputFormat(args['output_format'])
            except ValueError:
                pass  # Keep default
        
        if 'input_file' in args:
            self.input_file = args['input_file']
        
        if 'output_file' in args:
            self.output_file = args['output_file']
    
    def get_color(self, color_name: str) -> str:
        """Get color code for the specified color name."""
        if not self.display.use_colors:
            return ''
        return self.display.colors.get(color_name, '')
    
    def get_ascii_char(self, char_name: str) -> str:
        """Get ASCII character for the specified name."""
        if not self.display.use_unicode:
            # Fallback to basic ASCII
            fallbacks = {
                'tree_branch': '|-- ',
                'tree_last': '`-- ',
                'tree_vertical': '|   ',
                'arrow_right': '-> ',
                'arrow_down': 'v ',
                'bullet': '* '
            }
            return fallbacks.get(char_name, self.display.ascii_chars.get(char_name, ''))
        
        return self.display.ascii_chars.get(char_name, '')


def create_default_config() -> Config:
    """Create a default configuration instance."""
    return Config()


def load_config_from_file(filepath: str) -> Config:
    """Load configuration from a file (future enhancement)."""
    # For now, return default config
    # TODO: Implement file-based configuration loading
    return create_default_config()