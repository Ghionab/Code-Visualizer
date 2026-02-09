import sys
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import dis

from ..core.data_models import (
    TraceEvent, FrameState, AllocationInfo, MemoryLayout,
    ConditionalAnalysis, LoopAnalysis, CompilationStage,
    EnhancedInstruction, ExceptionAnalysis, IOOperation,
    FunctionCallInfo
)


class VerbosityLevel(Enum):
    """Verbosity levels for educational explanations."""
    MINIMAL = 1      # Only essential information
    NORMAL = 2       # Standard explanations
    DETAILED = 3     # Comprehensive explanations with context
    EDUCATIONAL = 4  # Full educational mode with "why" explanations


class ColorCode:
    """ANSI color codes for terminal output."""
    # Basic colors
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


@dataclass
class DisplayConfig:
    """Configuration for display engine."""
    use_colors: bool = True
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    max_line_width: int = 80
    indent_size: int = 2
    show_timestamps: bool = True
    show_memory_addresses: bool = True
    truncate_long_values: bool = True
    max_value_length: int = 50


class DisplayEngine:
   
    def __init__(self, config: Optional[DisplayConfig] = None):
        
        self.config = config or DisplayConfig()
        self.indent_level = 0
        self._source_lines: Dict[str, List[str]] = {}
        
        # Disable colors if not supported
        if not self._supports_color():
            self.config.use_colors = False
    
    def _supports_color(self) -> bool:
        """Check if the terminal supports color output."""
        # Check if stdout is a terminal
        if not hasattr(sys.stdout, 'isatty'):
            return False
        if not sys.stdout.isatty():
            return False
        
        # Check for Windows and ANSI support
        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable ANSI escape sequences on Windows 10+
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                return False
        
        return True
    
    def colorize(self, text: str, color: str) -> str:
       
        if self.config.use_colors:
            return f"{color}{text}{ColorCode.RESET}"
        return text
    
    def indent(self, text: str, level: Optional[int] = None) -> str:
       
        indent_level = level if level is not None else self.indent_level
        indent_str = ' ' * (indent_level * self.config.indent_size)
        
        # Handle multi-line text
        lines = text.split('\n')
        return '\n'.join(indent_str + line if line.strip() else line for line in lines)
    
    def truncate_value(self, value: Any, max_length: Optional[int] = None) -> str:
        
        max_len = max_length or self.config.max_value_length
        value_str = repr(value)
        
        if self.config.truncate_long_values and len(value_str) > max_len:
            return value_str[:max_len - 3] + "..."
        
        return value_str
    
    def format_header(self, title: str, level: int = 1) -> str:
        
        if level == 1:
            separator = "=" * self.config.max_line_width
            colored_title = self.colorize(title.upper(), ColorCode.BOLD + ColorCode.BRIGHT_CYAN)
            return f"\n{separator}\n{colored_title}\n{separator}\n"
        elif level == 2:
            separator = "-" * self.config.max_line_width
            colored_title = self.colorize(title, ColorCode.BOLD + ColorCode.CYAN)
            return f"\n{colored_title}\n{separator}\n"
        else:
            colored_title = self.colorize(title, ColorCode.BOLD + ColorCode.BLUE)
            return f"\n{colored_title}:\n"
    
    def format_compilation_stage(self, stage: CompilationStage) -> str:
       
        lines = []
        
        # Header
        lines.append(self.format_header(f"Compilation Stage: {stage.stage_name}", level=2))
        
        # Explanation
        if stage.explanation:
            explanation = self._format_explanation(stage.explanation, "Stage Overview")
            lines.append(self.indent(explanation))
            lines.append("")
        
        # Input data
        lines.append(self.indent(self.colorize("INPUT:", ColorCode.YELLOW)))
        input_repr = self._format_data_representation(stage.input_data, stage.stage_name)
        lines.append(self.indent(input_repr, self.indent_level + 1))
        lines.append("")
        
        # Output data
        lines.append(self.indent(self.colorize("OUTPUT:", ColorCode.GREEN)))
        output_repr = self._format_data_representation(stage.output_data, stage.stage_name)
        lines.append(self.indent(output_repr, self.indent_level + 1))
        
        # Metadata if available
        if stage.metadata:
            lines.append("")
            lines.append(self.indent(self.colorize("METADATA:", ColorCode.CYAN)))
            for key, value in stage.metadata.items():
                lines.append(self.indent(f"{key}: {value}", self.indent_level + 1))
        
        return '\n'.join(lines)
    
    def format_execution_step(self, event: TraceEvent, frame_state: Optional[FrameState] = None) -> str:
      
        lines = []
        
        # Timestamp if enabled
        timestamp_str = ""
        if self.config.show_timestamps:
            timestamp_str = self.colorize(f"[{event.timestamp:8.4f}s]", ColorCode.DIM)
        
        # Event type indicator
        event_color = self._get_event_color(event.event_type)
        event_indicator = self.colorize(f"{event.event_type.upper():10}", event_color)
        
        # Format based on event type
        if event.event_type == 'call':
            func_name = event.frame.f_code.co_name
            line_num = event.frame.f_lineno
            detail = f"→ {self.colorize(func_name, ColorCode.BRIGHT_YELLOW)}() at line {line_num}"
            
        elif event.event_type == 'line':
            line_num = event.frame.f_lineno
            func_name = event.frame.f_code.co_name
            detail = f"  Line {self.colorize(str(line_num), ColorCode.BRIGHT_WHITE)} in {func_name}()"
            
            # Add source line if available
            source_line = self._get_source_line(event.frame.f_code.co_filename, line_num)
            if source_line:
                detail += f"\n{self.indent(self.colorize(f'    {source_line.strip()}', ColorCode.DIM), self.indent_level + 1)}"
            
        elif event.event_type == 'return':
            func_name = event.frame.f_code.co_name
            return_value = self.truncate_value(event.arg)
            detail = f"← {self.colorize(func_name, ColorCode.BRIGHT_YELLOW)}() returns {self.colorize(return_value, ColorCode.GREEN)}"
            
        elif event.event_type == 'exception':
            exc_type = type(event.arg).__name__ if event.arg else "Exception"
            exc_msg = str(event.arg) if event.arg else ""
            detail = f"! {self.colorize(exc_type, ColorCode.BRIGHT_RED)}: {exc_msg}"
            
        else:
            detail = f"  {event.event_type}"
        
        # Combine components
        depth_indent = "  " * min(event.stack_depth, 10)
        line = f"{timestamp_str} {depth_indent}{event_indicator} {detail}"
        lines.append(self.indent(line))
        
        # Add frame state if provided and verbosity is high enough
        if frame_state and self.config.verbosity.value >= VerbosityLevel.DETAILED.value:
            frame_info = self._format_frame_state_compact(frame_state)
            lines.append(self.indent(frame_info, self.indent_level + 1))
        
        return '\n'.join(lines)

    def format_memory_layout(self, layout: MemoryLayout) -> str:

        lines = []
        
        # Header
        lines.append(self.format_header(f"Memory Layout: {layout.type_name}", level=3))
        
        # Basic information
        lines.append(self.indent(self.colorize("Object Information:", ColorCode.CYAN)))
        if self.config.show_memory_addresses:
            lines.append(self.indent(f"Address:  0x{layout.memory_address:016x}", self.indent_level + 1))
        lines.append(self.indent(f"Type:     {self.colorize(layout.type_name, ColorCode.YELLOW)}", self.indent_level + 1))
        lines.append(self.indent(f"Size:     {self.colorize(str(layout.size) + ' bytes', ColorCode.GREEN)}", self.indent_level + 1))
        lines.append(self.indent(f"RefCount: {self.colorize(str(layout.reference_count), ColorCode.MAGENTA)}", self.indent_level + 1))
        lines.append("")
        
        # ASCII diagram based on type
        diagram = self._create_memory_diagram(layout)
        lines.append(self.indent(diagram))
        
        # Internal structure if available
        if layout.internal_structure:
            lines.append("")
            lines.append(self.indent(self.colorize("Internal Structure:", ColorCode.CYAN)))
            for key, value in layout.internal_structure.items():
                lines.append(self.indent(f"{key}: {value}", self.indent_level + 1))
        
        return '\n'.join(lines)
    
    def format_stack_state(self, stack: List[Any], detailed: bool = False) -> str:

        lines = []
        
        lines.append(self.indent(self.colorize("Evaluation Stack:", ColorCode.CYAN)))
        
        if not stack:
            lines.append(self.indent(self.colorize("<empty>", ColorCode.DIM), self.indent_level + 1))
            return '\n'.join(lines)
        
        # Draw stack from top to bottom
        lines.append(self.indent("┌─────────────────────────────────────┐", self.indent_level + 1))
        
        for i, item in enumerate(reversed(stack)):
            item_repr = self.truncate_value(item, 30)
            item_type = type(item).__name__
            
            if detailed:
                line = f"│ [{len(stack)-1-i}] {item_repr:<25} │"
                type_line = f"│     ({item_type})                    │"
                lines.append(self.indent(self.colorize(line, ColorCode.YELLOW), self.indent_level + 1))
                lines.append(self.indent(self.colorize(type_line, ColorCode.DIM), self.indent_level + 1))
                if i < len(stack) - 1:
                    lines.append(self.indent("├─────────────────────────────────────┤", self.indent_level + 1))
            else:
                line = f"│ [{len(stack)-1-i}] {item_repr:<30} │"
                lines.append(self.indent(self.colorize(line, ColorCode.YELLOW), self.indent_level + 1))
        
        lines.append(self.indent("└─────────────────────────────────────┘", self.indent_level + 1))
        lines.append(self.indent(self.colorize(f"(depth: {len(stack)})", ColorCode.DIM), self.indent_level + 1))
        
        return '\n'.join(lines)
    
    def format_control_flow(self, analysis: Union[ConditionalAnalysis, LoopAnalysis]) -> str:

        lines = []
        
        if isinstance(analysis, ConditionalAnalysis):
            lines.append(self.indent(self.colorize("Conditional Branch:", ColorCode.MAGENTA)))
            lines.append(self.indent(f"Condition: {self.truncate_value(analysis.condition_value)}", self.indent_level + 1))
            
            branch_color = ColorCode.GREEN if analysis.branch_taken == 'true' else ColorCode.RED
            lines.append(self.indent(f"Branch: {self.colorize(analysis.branch_taken.upper(), branch_color)}", self.indent_level + 1))
            
            if analysis.short_circuit:
                lines.append(self.indent(self.colorize("⚡ Short-circuit evaluation", ColorCode.YELLOW), self.indent_level + 1))
            
            if analysis.source_line:
                lines.append(self.indent(f"Source: {self.colorize(analysis.source_line.strip(), ColorCode.DIM)}", self.indent_level + 1))
            
            # Educational explanation
            if self.config.verbosity.value >= VerbosityLevel.NORMAL.value:
                lines.append("")
                lines.append(self.indent(self._format_explanation(analysis.explanation, "Why"), self.indent_level + 1))
        
        elif isinstance(analysis, LoopAnalysis):
            lines.append(self.indent(self.colorize(f"Loop Iteration ({analysis.loop_type}):", ColorCode.MAGENTA)))
            lines.append(self.indent(f"Iterator: {self.truncate_value(analysis.iterator_object)}", self.indent_level + 1))
            lines.append(self.indent(f"Current Value: {self.colorize(self.truncate_value(analysis.current_value), ColorCode.GREEN)}", self.indent_level + 1))
            lines.append(self.indent(f"Iteration: {self.colorize(str(analysis.iteration_count), ColorCode.CYAN)}", self.indent_level + 1))
            
            if analysis.is_exhausted:
                lines.append(self.indent(self.colorize("✓ Iterator exhausted", ColorCode.YELLOW), self.indent_level + 1))
            
            # Educational explanation
            if self.config.verbosity.value >= VerbosityLevel.NORMAL.value:
                lines.append("")
                lines.append(self.indent(self._format_explanation(analysis.explanation, "How"), self.indent_level + 1))
        
        return '\n'.join(lines)
    
    def format_bytecode_instruction(self, instruction: Union[dis.Instruction, EnhancedInstruction]) -> str:

        lines = []
        
        # Extract instruction details
        if isinstance(instruction, EnhancedInstruction):
            instr = instruction.instruction
            explanation = instruction.explanation
            source_line = instruction.source_line
            stack_effect = instruction.stack_effect
        else:
            instr = instruction
            explanation = None
            source_line = None
            stack_effect = None
        
        # Format instruction
        offset_str = self.colorize(f"{instr.offset:4d}", ColorCode.DIM)
        opname_str = self.colorize(f"{instr.opname:<20}", ColorCode.BRIGHT_YELLOW)
        
        arg_str = ""
        if instr.arg is not None:
            arg_str = f"{instr.arg:4d}"
            if instr.argval is not None:
                argval_repr = self.truncate_value(instr.argval, 30)
                arg_str += f" ({argval_repr})"
        
        line = f"{offset_str} {opname_str} {arg_str}"
        lines.append(self.indent(line))
        
        # Stack effect
        if stack_effect is not None:
            effect_color = ColorCode.GREEN if stack_effect >= 0 else ColorCode.RED
            effect_str = self.colorize(f"Stack: {stack_effect:+d}", effect_color)
            lines.append(self.indent(f"  {effect_str}", self.indent_level + 1))
        
        # Source line mapping
        if source_line:
            lines.append(self.indent(f"  Source: {self.colorize(source_line.strip(), ColorCode.DIM)}", self.indent_level + 1))
        
        # Explanation if available and verbosity allows
        if explanation and self.config.verbosity.value >= VerbosityLevel.NORMAL.value:
            lines.append(self.indent(f"  → {self.colorize(explanation, ColorCode.CYAN)}", self.indent_level + 1))
        
        return '\n'.join(lines)
    
    def format_exception(self, analysis: ExceptionAnalysis) -> str:

        lines = []
        
        lines.append(self.format_header("Exception Raised", level=3))
        
        # Exception type and value
        exc_type_str = self.colorize(analysis.exception_type.__name__, ColorCode.BRIGHT_RED + ColorCode.BOLD)
        lines.append(self.indent(f"Type: {exc_type_str}"))
        lines.append(self.indent(f"Value: {self.colorize(str(analysis.exception_value), ColorCode.RED)}"))
        lines.append("")
        
        # Handler status
        if analysis.handler_found:
            lines.append(self.indent(self.colorize("✓ Exception handler found", ColorCode.GREEN)))
        else:
            lines.append(self.indent(self.colorize("✗ No handler found - will propagate", ColorCode.YELLOW)))
        lines.append("")
        
        # Propagation path
        if analysis.propagation_path:
            lines.append(self.indent(self.colorize("Propagation Path:", ColorCode.CYAN)))
            for i, frame_name in enumerate(analysis.propagation_path):
                arrow = "  " * i + "↑ "
                lines.append(self.indent(f"{arrow}{frame_name}", self.indent_level + 1))
            lines.append("")
        
        # Traceback information
        if analysis.traceback_info:
            lines.append(self.indent(self.colorize("Traceback:", ColorCode.CYAN)))
            for tb_entry in analysis.traceback_info:
                filename = tb_entry.get('filename', '<unknown>')
                lineno = tb_entry.get('lineno', 0)
                func_name = tb_entry.get('name', '<unknown>')
                lines.append(self.indent(f"  File \"{filename}\", line {lineno}, in {func_name}", self.indent_level + 1))
        
        # Educational explanation
        if self.config.verbosity.value >= VerbosityLevel.NORMAL.value:
            lines.append("")
            lines.append(self.indent(self._format_explanation(analysis.explanation, "What Happened")))
        
        return '\n'.join(lines)
    
    def format_function_call(self, call_info: FunctionCallInfo) -> str:

        lines = []
        
        # Function name with call depth indicator
        depth_indicator = "  " * call_info.call_depth
        func_name = self.colorize(call_info.function_name, ColorCode.BRIGHT_YELLOW + ColorCode.BOLD)
        lines.append(self.indent(f"{depth_indicator}→ Calling {func_name}()"))
        
        # Arguments
        if call_info.arguments:
            lines.append(self.indent(self.colorize("Arguments:", ColorCode.CYAN), self.indent_level + 1))
            for arg_name, arg_value in call_info.arguments.items():
                arg_repr = self.truncate_value(arg_value)
                lines.append(self.indent(f"{arg_name} = {arg_repr}", self.indent_level + 2))
        
        # Closure variables if present
        if call_info.closure_vars:
            lines.append(self.indent(self.colorize("Closure Variables:", ColorCode.MAGENTA), self.indent_level + 1))
            for var_name, var_value in call_info.closure_vars.items():
                var_repr = self.truncate_value(var_value)
                lines.append(self.indent(f"{var_name} = {var_repr}", self.indent_level + 2))
        
        # Local scope (if verbosity is high)
        if self.config.verbosity.value >= VerbosityLevel.DETAILED.value and call_info.local_scope:
            lines.append(self.indent(self.colorize("Local Scope:", ColorCode.CYAN), self.indent_level + 1))
            for var_name, var_value in list(call_info.local_scope.items())[:5]:
                if not var_name.startswith('__'):
                    var_repr = self.truncate_value(var_value)
                    lines.append(self.indent(f"{var_name} = {var_repr}", self.indent_level + 2))
        
        # Return value if available
        if call_info.return_value is not None:
            return_repr = self.truncate_value(call_info.return_value)
            lines.append(self.indent(f"{depth_indicator}← Returns: {self.colorize(return_repr, ColorCode.GREEN)}", self.indent_level))
        
        return '\n'.join(lines)
    
    def format_io_operation(self, io_op: IOOperation) -> str:
        """
        Format I/O operation information.
        
        Args:
            io_op: IOOperation object
            
        Returns:
            Formatted string showing I/O operation details
        """
        lines = []
        
        # Operation type with icon
        op_icons = {
            'read': '📖',
            'write': '✍️',
            'open': '📂',
            'close': '🔒'
        }
        icon = op_icons.get(io_op.operation_type, '💾')
        op_color = ColorCode.BLUE if io_op.operation_type in ['read', 'open'] else ColorCode.GREEN
        
        lines.append(self.indent(f"{icon} {self.colorize(io_op.operation_type.upper(), op_color)} Operation"))
        
        # Target
        lines.append(self.indent(f"Target: {self.colorize(io_op.target, ColorCode.YELLOW)}", self.indent_level + 1))
        
        # Data (truncated for large data)
        if io_op.data is not None:
            data_repr = self.truncate_value(io_op.data, 60)
            lines.append(self.indent(f"Data: {data_repr}", self.indent_level + 1))
        
        # Result
        if io_op.result is not None:
            result_repr = self.truncate_value(io_op.result, 60)
            lines.append(self.indent(f"Result: {self.colorize(result_repr, ColorCode.GREEN)}", self.indent_level + 1))
        
        # System call if available
        if io_op.system_call:
            lines.append(self.indent(f"System Call: {self.colorize(io_op.system_call, ColorCode.MAGENTA)}", self.indent_level + 1))
        
        # Timestamp
        if self.config.show_timestamps:
            lines.append(self.indent(self.colorize(f"Time: {io_op.timestamp:.4f}s", ColorCode.DIM), self.indent_level + 1))
        
        return '\n'.join(lines)
    
    # Helper methods for internal use
    
    def _get_event_color(self, event_type: str) -> str:
        """Get color code for event type."""
        color_map = {
            'call': ColorCode.BRIGHT_GREEN,
            'line': ColorCode.BRIGHT_BLUE,
            'return': ColorCode.BRIGHT_MAGENTA,
            'exception': ColorCode.BRIGHT_RED,
            'opcode': ColorCode.BRIGHT_YELLOW
        }
        return color_map.get(event_type, ColorCode.WHITE)
    
    def _format_data_representation(self, data: Any, context: str) -> str:
        """Format data representation based on context."""
        if context == 'tokenize':
            # Format token stream
            if isinstance(data, list):
                lines = []
                for i, token in enumerate(data[:20]):  # Show first 20 tokens
                    token_repr = self.truncate_value(token, 60)
                    lines.append(f"[{i}] {token_repr}")
                if len(data) > 20:
                    lines.append(f"... and {len(data) - 20} more tokens")
                return '\n'.join(lines)
        
        elif context == 'parse':
            # Format AST representation
            return self._format_ast_tree(data)
        
        elif context == 'compile':
            # Format bytecode
            if hasattr(data, 'co_code'):
                return f"Code object: {data.co_name} ({len(data.co_code)} bytes)"
        
        # Default representation
        return self.truncate_value(data, 100)
    
    def _format_ast_tree(self, node: Any, depth: int = 0) -> str:
        """Format AST tree structure."""
        if not hasattr(node, '__class__'):
            return str(node)
        
        lines = []
        indent = "  " * depth
        node_name = node.__class__.__name__
        lines.append(f"{indent}{self.colorize(node_name, ColorCode.YELLOW)}")
        
        # Limit depth for display
        if depth < 3 and hasattr(node, '_fields'):
            for field in node._fields[:5]:  # Show first 5 fields
                value = getattr(node, field, None)
                if value is not None:
                    if isinstance(value, list):
                        lines.append(f"{indent}  {field}: [{len(value)} items]")
                    elif hasattr(value, '__class__') and hasattr(value, '_fields'):
                        lines.append(f"{indent}  {field}:")
                        lines.append(self._format_ast_tree(value, depth + 2))
                    else:
                        lines.append(f"{indent}  {field}: {self.truncate_value(value, 30)}")
        
        return '\n'.join(lines)
    
    def _create_memory_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for memory layout."""
        # Delegate to type-specific diagram creators
        if layout.type_name == 'list':
            return self._create_list_diagram(layout)
        elif layout.type_name == 'dict':
            return self._create_dict_diagram(layout)
        elif layout.type_name == 'tuple':
            return self._create_tuple_diagram(layout)
        elif layout.type_name == 'set':
            return self._create_set_diagram(layout)
        else:
            return self._create_generic_diagram(layout)
    
    def _create_list_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for list objects."""
        lines = []
        lines.append("┌─────────────────────────────────┐")
        lines.append(f"│ List Object                     │")
        lines.append("├─────────────────────────────────┤")
        
        # Show fields (list items)
        items = [(k, v) for k, v in layout.fields.items() if k.startswith('item_')]
        if items:
            for key, value in items[:5]:
                idx = key.split('_')[1]
                value_repr = self.truncate_value(value, 25)
                lines.append(f"│ [{idx}] → {value_repr:<25} │")
            if len(items) > 5:
                lines.append(f"│ ... ({len(items) - 5} more items)       │")
        else:
            lines.append("│ (empty)                         │")
        
        lines.append("└─────────────────────────────────┘")
        
        return '\n'.join(lines)
    
    def _create_dict_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for dict objects."""
        lines = []
        lines.append("┌─────────────────────────────────────────┐")
        lines.append(f"│ Dictionary Object                       │")
        lines.append("├─────────────────────────────────────────┤")
        
        # Show key-value pairs
        items = [(k, v) for k, v in layout.fields.items() if k.startswith('key_')]
        if items:
            for key, value in items[:5]:
                if isinstance(value, tuple) and len(value) == 2:
                    k, v = value
                    key_repr = self.truncate_value(k, 15)
                    val_repr = self.truncate_value(v, 20)
                    lines.append(f"│ {key_repr:<15} : {val_repr:<20} │")
            if len(items) > 5:
                lines.append(f"│ ... ({len(items) - 5} more items)           │")
        else:
            lines.append("│ (empty)                                 │")
        
        lines.append("└─────────────────────────────────────────┘")
        
        return '\n'.join(lines)
    
    def _create_tuple_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for tuple objects."""
        lines = []
        lines.append("┌─────────────────────────────────┐")
        lines.append(f"│ Tuple Object (Immutable)        │")
        lines.append("├─────────────────────────────────┤")
        
        # Show items
        items = [(k, v) for k, v in layout.fields.items() if k.startswith('item_')]
        if items:
            for key, value in items[:5]:
                idx = key.split('_')[1]
                value_repr = self.truncate_value(value, 25)
                lines.append(f"│ ({idx}) → {value_repr:<25} │")
            if len(items) > 5:
                lines.append(f"│ ... ({len(items) - 5} more items)       │")
        else:
            lines.append("│ (empty)                         │")
        
        lines.append("└─────────────────────────────────┘")
        
        return '\n'.join(lines)
    
    def _create_set_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for set objects."""
        lines = []
        lines.append("┌─────────────────────────────────┐")
        lines.append(f"│ Set Object                      │")
        lines.append("├─────────────────────────────────┤")
        
        # Show items
        items = [(k, v) for k, v in layout.fields.items() if k.startswith('item_')]
        if items:
            for key, value in items[:5]:
                idx = key.split('_')[1]
                value_repr = self.truncate_value(value, 25)
                lines.append(f"│ {{{idx}}} → {value_repr:<25} │")
            if len(items) > 5:
                lines.append(f"│ ... ({len(items) - 5} more items)       │")
        else:
            lines.append("│ (empty)                         │")
        
        lines.append("└─────────────────────────────────┘")
        
        return '\n'.join(lines)
    
    def _create_generic_diagram(self, layout: MemoryLayout) -> str:
        """Create ASCII diagram for generic objects."""
        lines = []
        lines.append("┌─────────────────────────────────────────┐")
        lines.append(f"│ {layout.type_name} Object                │")
        lines.append("├─────────────────────────────────────────┤")
        
        if layout.fields:
            for key, value in list(layout.fields.items())[:10]:
                key_repr = str(key)[:15]
                val_repr = self.truncate_value(value, 20)
                lines.append(f"│ {key_repr:<15} : {val_repr:<20} │")
            if len(layout.fields) > 10:
                lines.append(f"│ ... ({len(layout.fields) - 10} more fields)      │")
        else:
            value_repr = self.truncate_value(layout.fields.get('value', ''), 35)
            lines.append(f"│ Value: {value_repr:<35} │")
        
        lines.append("└─────────────────────────────────────────┘")
        
        return '\n'.join(lines)
    
    def _format_explanation(self, explanation: str, title: str = "Explanation") -> str:
        """Format an educational explanation."""
        lines = []
        
        # Title
        title_str = self.colorize(f"💡 {title}:", ColorCode.BRIGHT_CYAN)
        lines.append(title_str)
        
        # Wrap explanation text
        wrapped = self._wrap_text(explanation, self.config.max_line_width - (self.indent_level + 1) * self.config.indent_size)
        for line in wrapped:
            lines.append(f"   {line}")
        
        return '\n'.join(lines)
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + len(current_line) <= width:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _get_source_line(self, filename: str, line_number: int) -> Optional[str]:
        """Get source line from file."""
        # Cache source lines
        if filename not in self._source_lines:
            try:
                with open(filename, 'r') as f:
                    self._source_lines[filename] = f.readlines()
            except:
                return None
        
        lines = self._source_lines.get(filename, [])
        if 0 < line_number <= len(lines):
            return lines[line_number - 1]
        
        return None
    
    def _format_frame_state_compact(self, frame_state: FrameState) -> str:
        """Format frame state in compact form."""
        lines = []
        
        # Show only non-empty locals
        if frame_state.locals:
            local_vars = {k: v for k, v in frame_state.locals.items() if not k.startswith('__')}
            if local_vars:
                vars_str = ', '.join(f"{k}={self.truncate_value(v, 15)}" for k, v in list(local_vars.items())[:3])
                lines.append(self.colorize(f"Locals: {vars_str}", ColorCode.DIM))
        
        return '\n'.join(lines)
    
    def set_verbosity(self, level: VerbosityLevel):
        """Set the verbosity level for explanations."""
        self.config.verbosity = level
    
    def enable_colors(self, enabled: bool):
        """Enable or disable color output."""
        self.config.use_colors = enabled
    
    def push_indent(self):
        """Increase indentation level."""
        self.indent_level += 1
    
    def pop_indent(self):
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)
    
    def reset_indent(self):
        """Reset indentation to zero."""
        self.indent_level = 0



class EducationalExplainer:
    """
    Provides educational explanations for Python execution concepts.
    
    Generates plain-English explanations with varying levels of detail,
    including "why" explanations and source code context.
    """
    
    def __init__(self, verbosity: VerbosityLevel = VerbosityLevel.NORMAL):
        """
        Initialize the educational explainer.
        
        Args:
            verbosity: Level of detail for explanations
        """
        self.verbosity = verbosity
        self._opcode_explanations = self._build_opcode_explanations()
        self._concept_explanations = self._build_concept_explanations()
    
    def explain_opcode(self, instruction: dis.Instruction, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate plain-English explanation for a bytecode instruction.
        
        Args:
            instruction: Bytecode instruction to explain
            context: Optional context (locals, globals, stack state)
            
        Returns:
            Human-readable explanation
        """
        opname = instruction.opname
        base_explanation = self._opcode_explanations.get(opname, f"Execute {opname} operation")
        
        # Customize explanation based on verbosity
        if self.verbosity == VerbosityLevel.MINIMAL:
            return base_explanation.split('.')[0]  # First sentence only
        
        # Add context-specific details
        if context and self.verbosity.value >= VerbosityLevel.NORMAL.value:
            explanation = self._contextualize_opcode(instruction, base_explanation, context)
        else:
            explanation = base_explanation
        
        # Add "why" explanation for educational mode
        if self.verbosity == VerbosityLevel.EDUCATIONAL:
            why_explanation = self._explain_why_opcode(instruction)
            if why_explanation:
                explanation += f"\n\n💡 Why: {why_explanation}"
        
        return explanation
    
    def explain_control_flow(self, flow_type: str, details: Dict[str, Any]) -> str:
        """
        Generate explanation for control flow constructs.
        
        Args:
            flow_type: Type of control flow ('if', 'for', 'while', 'try', etc.)
            details: Details about the control flow
            
        Returns:
            Educational explanation
        """
        base_explanation = self._concept_explanations.get(f"control_flow_{flow_type}", "")
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            return base_explanation.split('.')[0]
        
        # Add specific details
        explanation = base_explanation
        
        if flow_type == 'conditional':
            condition = details.get('condition')
            branch = details.get('branch_taken')
            explanation += f" The condition evaluated to {condition}, so the {branch} branch was taken."
            
            if self.verbosity == VerbosityLevel.EDUCATIONAL:
                explanation += "\n\n💡 Why: Python evaluates conditions to determine which code path to execute. This allows your program to make decisions based on data."
        
        elif flow_type == 'loop':
            iteration = details.get('iteration_count', 0)
            explanation += f" This is iteration {iteration} of the loop."
            
            if self.verbosity == VerbosityLevel.EDUCATIONAL:
                explanation += "\n\n💡 Why: Loops allow you to repeat code multiple times without writing it repeatedly. Python's for loops use the iterator protocol to fetch values one at a time."
        
        return explanation
    
    def explain_memory_operation(self, operation: str, obj_type: str, details: Dict[str, Any]) -> str:
        """
        Generate explanation for memory operations.
        
        Args:
            operation: Type of operation ('allocate', 'mutate', 'deallocate')
            obj_type: Type of object involved
            details: Operation details
            
        Returns:
            Educational explanation
        """
        explanations = {
            'allocate': f"A new {obj_type} object is being created in memory.",
            'mutate': f"The {obj_type} object is being modified.",
            'deallocate': f"The {obj_type} object is being removed from memory."
        }
        
        base_explanation = explanations.get(operation, f"Memory operation on {obj_type}")
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            return base_explanation
        
        # Add details based on object type
        if operation == 'allocate':
            size = details.get('size', 0)
            base_explanation += f" It will occupy {size} bytes of memory."
            
            if obj_type == 'list' and self.verbosity.value >= VerbosityLevel.DETAILED.value:
                base_explanation += " Lists in Python are dynamic arrays that can grow and shrink as needed."
            elif obj_type == 'dict' and self.verbosity.value >= VerbosityLevel.DETAILED.value:
                base_explanation += " Dictionaries use hash tables for fast key-value lookups."
        
        elif operation == 'mutate':
            if obj_type == 'list':
                base_explanation += " Lists are mutable, so their contents can be changed after creation."
            elif obj_type == 'tuple':
                base_explanation += " Note: Tuples are immutable, so this operation creates a new tuple."
        
        # Add "why" for educational mode
        if self.verbosity == VerbosityLevel.EDUCATIONAL:
            why_explanations = {
                'allocate': "Memory allocation is necessary to store data during program execution. Python's memory manager handles this automatically.",
                'mutate': "Mutation allows you to update data structures efficiently without creating new objects. However, it can make code harder to reason about.",
                'deallocate': "Python uses garbage collection to automatically free memory when objects are no longer needed, preventing memory leaks."
            }
            why = why_explanations.get(operation, "")
            if why:
                base_explanation += f"\n\n💡 Why: {why}"
        
        return base_explanation
    
    def explain_function_call(self, func_name: str, details: Dict[str, Any]) -> str:
        """
        Generate explanation for function calls.
        
        Args:
            func_name: Name of the function being called
            details: Call details (arguments, closure, etc.)
            
        Returns:
            Educational explanation
        """
        explanation = f"Calling function '{func_name}'."
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            return explanation
        
        # Add argument information
        args = details.get('arguments', {})
        if args:
            arg_count = len(args)
            explanation += f" The function receives {arg_count} argument{'s' if arg_count != 1 else ''}."
        
        # Add closure information
        closure_vars = details.get('closure_vars', {})
        if closure_vars and self.verbosity.value >= VerbosityLevel.DETAILED.value:
            explanation += f" This function captures {len(closure_vars)} variable{'s' if len(closure_vars) != 1 else ''} from its enclosing scope."
        
        # Add frame information
        if self.verbosity.value >= VerbosityLevel.DETAILED.value:
            explanation += " Python creates a new execution frame with its own local namespace."
        
        # Educational explanation
        if self.verbosity == VerbosityLevel.EDUCATIONAL:
            explanation += "\n\n💡 Why: Functions allow you to organize code into reusable pieces. Each function call creates a new frame on the call stack, which stores the function's local variables and execution state."
            
            if closure_vars:
                explanation += " Closures allow functions to 'remember' variables from their defining scope, enabling powerful patterns like decorators and callbacks."
        
        return explanation
    
    def explain_exception(self, exc_type: str, details: Dict[str, Any]) -> str:
        """
        Generate explanation for exceptions.
        
        Args:
            exc_type: Type of exception
            details: Exception details
            
        Returns:
            Educational explanation
        """
        explanation = f"A {exc_type} exception has been raised."
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            return explanation
        
        # Add context about what caused it
        message = details.get('message', '')
        if message:
            explanation += f" {message}"
        
        # Add propagation information
        handler_found = details.get('handler_found', False)
        if handler_found:
            explanation += " Python will look for an exception handler (try/except block) to handle this error."
        else:
            explanation += " No exception handler was found, so the exception will propagate up the call stack."
        
        # Educational explanation
        if self.verbosity == VerbosityLevel.EDUCATIONAL:
            explanation += "\n\n💡 Why: Exceptions are Python's way of handling errors and exceptional conditions. They allow you to separate error-handling code from normal code flow, making programs more robust and easier to maintain."
            
            if handler_found:
                explanation += " When an exception is caught, the program can recover gracefully instead of crashing."
            else:
                explanation += " Unhandled exceptions will terminate the program and display a traceback to help you debug the issue."
        
        return explanation
    
    def explain_source_context(self, source_line: str, line_number: int, 
                              operation: str) -> str:
        """
        Generate explanation connecting bytecode to source code.
        
        Args:
            source_line: The source code line
            line_number: Line number in source file
            operation: What operation is being performed
            
        Returns:
            Explanation with source context
        """
        explanation = f"At line {line_number}: `{source_line.strip()}`"
        
        if self.verbosity == VerbosityLevel.MINIMAL:
            return explanation
        
        explanation += f"\n{operation}"
        
        if self.verbosity.value >= VerbosityLevel.DETAILED.value:
            explanation += f"\n\nThis source line compiles to bytecode instructions that the Python interpreter executes."
        
        if self.verbosity == VerbosityLevel.EDUCATIONAL:
            explanation += " Understanding how source code maps to bytecode helps you understand Python's execution model and can help you write more efficient code."
        
        return explanation
    
    def _build_opcode_explanations(self) -> Dict[str, str]:
        """Build dictionary of opcode explanations."""
        return {
            # Load operations
            'LOAD_CONST': "Load a constant value onto the stack. Constants are values that don't change, like numbers or strings defined in your code.",
            'LOAD_FAST': "Load a local variable onto the stack. This is the fastest way to access variables in the current function.",
            'LOAD_GLOBAL': "Load a global variable onto the stack. Global variables are defined at the module level.",
            'LOAD_NAME': "Load a variable by name. Python searches local, then global, then built-in namespaces.",
            'LOAD_ATTR': "Load an attribute from an object. This is what happens when you use dot notation like obj.attribute.",
            'LOAD_METHOD': "Load a method from an object for calling. This is optimized for method calls.",
            
            # Store operations
            'STORE_FAST': "Store the top stack value into a local variable. This is how assignment to local variables works.",
            'STORE_GLOBAL': "Store the top stack value into a global variable.",
            'STORE_NAME': "Store the top stack value into a variable by name.",
            'STORE_ATTR': "Store a value into an object's attribute. This is what happens during attribute assignment.",
            
            # Arithmetic operations
            'BINARY_ADD': "Add the top two stack values. This implements the + operator.",
            'BINARY_SUBTRACT': "Subtract the top stack value from the second. This implements the - operator.",
            'BINARY_MULTIPLY': "Multiply the top two stack values. This implements the * operator.",
            'BINARY_TRUE_DIVIDE': "Divide the second stack value by the top value. This implements the / operator.",
            'BINARY_FLOOR_DIVIDE': "Perform floor division. This implements the // operator.",
            'BINARY_MODULO': "Calculate the remainder. This implements the % operator.",
            'BINARY_POWER': "Raise the second value to the power of the top value. This implements the ** operator.",
            
            # Comparison operations
            'COMPARE_OP': "Compare two values using the specified comparison operator (<, >, ==, !=, etc.).",
            
            # Control flow
            'POP_JUMP_IF_FALSE': "Pop the top stack value and jump to a different instruction if it's false. This implements conditional branching.",
            'POP_JUMP_IF_TRUE': "Pop the top stack value and jump if it's true.",
            'JUMP_FORWARD': "Unconditionally jump forward to a different instruction.",
            'JUMP_ABSOLUTE': "Unconditionally jump to a specific instruction.",
            
            # Function calls
            'CALL_FUNCTION': "Call a function with the specified number of arguments. Arguments are popped from the stack.",
            'CALL_METHOD': "Call a method on an object. This is optimized for method calls.",
            'RETURN_VALUE': "Return from the current function with the top stack value as the return value.",
            
            # Stack manipulation
            'POP_TOP': "Remove and discard the top stack value. This is used when a value is no longer needed.",
            'DUP_TOP': "Duplicate the top stack value. This is useful when you need to use a value multiple times.",
            'ROT_TWO': "Swap the top two stack values.",
            'ROT_THREE': "Rotate the top three stack values.",
            
            # Collection operations
            'BUILD_LIST': "Create a new list from stack values.",
            'BUILD_TUPLE': "Create a new tuple from stack values.",
            'BUILD_MAP': "Create a new dictionary from stack values.",
            'BUILD_SET': "Create a new set from stack values.",
            
            # Iteration
            'GET_ITER': "Get an iterator from an object. This is the first step in a for loop.",
            'FOR_ITER': "Get the next value from an iterator. If the iterator is exhausted, jump forward.",
            
            # Exception handling
            'SETUP_EXCEPT': "Set up an exception handler. This marks the beginning of a try block.",
            'POP_EXCEPT': "Clean up after handling an exception.",
            'RAISE_VARARGS': "Raise an exception.",
            
            # Other operations
            'MAKE_FUNCTION': "Create a new function object from code on the stack.",
            'LOAD_BUILD_CLASS': "Load the __build_class__ function for creating classes.",
            'IMPORT_NAME': "Import a module by name.",
            'IMPORT_FROM': "Import a specific name from a module.",
        }
    
    def _build_concept_explanations(self) -> Dict[str, str]:
        """Build dictionary of concept explanations."""
        return {
            'control_flow_conditional': "Python evaluates a conditional expression to decide which code path to execute.",
            'control_flow_loop': "Python uses the iterator protocol to repeatedly execute a block of code.",
            'control_flow_exception': "Python uses exceptions to handle errors and exceptional conditions.",
            'memory_allocation': "Python allocates memory to store objects and data structures.",
            'memory_reference_counting': "Python tracks how many references point to each object to manage memory.",
            'function_call': "Python creates a new execution frame for each function call.",
            'closure': "A closure is a function that captures variables from its enclosing scope.",
        }
    
    def _contextualize_opcode(self, instruction: dis.Instruction, 
                             base_explanation: str, context: Dict[str, Any]) -> str:
        """Add context-specific details to opcode explanation."""
        opname = instruction.opname
        
        # Add value information for load operations
        if opname.startswith('LOAD_') and instruction.argval is not None:
            value_repr = repr(instruction.argval)
            if len(value_repr) > 30:
                value_repr = value_repr[:27] + "..."
            base_explanation += f" The value is: {value_repr}"
        
        # Add variable name for store operations
        elif opname.startswith('STORE_') and instruction.argval is not None:
            base_explanation += f" The variable name is '{instruction.argval}'."
        
        # Add operator information for comparisons
        elif opname == 'COMPARE_OP' and instruction.argval is not None:
            base_explanation += f" The comparison operator is '{instruction.argval}'."
        
        # Add function name for calls
        elif opname in ['CALL_FUNCTION', 'CALL_METHOD']:
            # Try to get function name from context
            func_name = context.get('function_name', 'unknown')
            if func_name != 'unknown':
                base_explanation += f" Calling '{func_name}'."
        
        return base_explanation
    
    def _explain_why_opcode(self, instruction: dis.Instruction) -> Optional[str]:
        """Generate 'why' explanation for an opcode."""
        opname = instruction.opname
        
        why_explanations = {
            'LOAD_FAST': "Local variables are stored in a fixed-size array, making access very fast. This is why you should prefer local variables for frequently accessed data.",
            'LOAD_GLOBAL': "Global variables require a dictionary lookup, which is slower than local variable access. This is one reason to minimize global variable usage.",
            'BINARY_ADD': "Python's + operator is polymorphic - it works differently for numbers, strings, lists, etc. The interpreter determines the correct operation at runtime.",
            'CALL_FUNCTION': "Function calls have overhead (creating a frame, copying arguments, etc.), which is why inlining or avoiding unnecessary calls can improve performance.",
            'FOR_ITER': "Python's iterator protocol allows any object to be iterable by implementing __iter__ and __next__. This makes for loops very flexible.",
            'POP_JUMP_IF_FALSE': "Conditional jumps are how Python implements if statements and short-circuit evaluation. The bytecode is structured to skip code that shouldn't execute.",
        }
        
        return why_explanations.get(opname)
    
    def set_verbosity(self, level: VerbosityLevel):
        """Set the verbosity level."""
        self.verbosity = level


class SourceContextMapper:
    """
    Maps bytecode instructions and execution events back to source code.
    
    Provides context about which source code lines correspond to which
    bytecode instructions and execution events.
    """
    
    def __init__(self):
        """Initialize the source context mapper."""
        self._source_cache: Dict[str, List[str]] = {}
        self._bytecode_to_source: Dict[int, int] = {}
    
    def load_source(self, filename: str) -> bool:
        """
        Load source code from a file.
        
        Args:
            filename: Path to source file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self._source_cache[filename] = f.readlines()
            return True
        except Exception:
            return False
    
    def get_source_line(self, filename: str, line_number: int) -> Optional[str]:
        """
        Get a specific source line.
        
        Args:
            filename: Source file path
            line_number: Line number (1-indexed)
            
        Returns:
            Source line or None if not available
        """
        if filename not in self._source_cache:
            if not self.load_source(filename):
                return None
        
        lines = self._source_cache.get(filename, [])
        if 0 < line_number <= len(lines):
            return lines[line_number - 1]
        
        return None
    
    def get_source_context(self, filename: str, line_number: int, 
                          context_lines: int = 2) -> Optional[List[tuple[int, str]]]:
        """
        Get source lines with context around a specific line.
        
        Args:
            filename: Source file path
            line_number: Target line number
            context_lines: Number of lines before and after to include
            
        Returns:
            List of (line_number, line_text) tuples or None
        """
        if filename not in self._source_cache:
            if not self.load_source(filename):
                return None
        
        lines = self._source_cache.get(filename, [])
        start = max(1, line_number - context_lines)
        end = min(len(lines), line_number + context_lines)
        
        result = []
        for i in range(start, end + 1):
            if 0 < i <= len(lines):
                result.append((i, lines[i - 1]))
        
        return result
    
    def map_instruction_to_source(self, instruction: dis.Instruction, 
                                 code_object: Any) -> Optional[int]:
        """
        Map a bytecode instruction to its source line.
        
        Args:
            instruction: Bytecode instruction
            code_object: Code object containing the instruction
            
        Returns:
            Source line number or None
        """
        # Use the instruction's starts_line attribute if available
        if hasattr(instruction, 'starts_line') and instruction.starts_line is not None:
            return instruction.starts_line
        
        # Fall back to code object's line number table
        if hasattr(code_object, 'co_lnotab'):
            # This is a simplified version - real implementation would parse co_lnotab
            return None
        
        return None
    
    def format_source_with_highlight(self, filename: str, line_number: int,
                                    context_lines: int = 2, 
                                    display_engine: Optional[DisplayEngine] = None) -> str:
        """
        Format source code with the target line highlighted.
        
        Args:
            filename: Source file path
            line_number: Line to highlight
            context_lines: Context lines to show
            display_engine: Optional DisplayEngine for formatting
            
        Returns:
            Formatted source code with highlighting
        """
        context = self.get_source_context(filename, line_number, context_lines)
        if not context:
            return f"Source not available for {filename}:{line_number}"
        
        lines = []
        for num, text in context:
            # Format line number
            line_num_str = f"{num:4d} "
            
            # Highlight the target line
            if num == line_number:
                if display_engine:
                    line_str = display_engine.colorize(f"{line_num_str}→ {text.rstrip()}", 
                                                       ColorCode.BRIGHT_YELLOW + ColorCode.BOLD)
                else:
                    line_str = f"{line_num_str}→ {text.rstrip()}"
            else:
                if display_engine:
                    line_str = display_engine.colorize(f"{line_num_str}  {text.rstrip()}", 
                                                       ColorCode.DIM)
                else:
                    line_str = f"{line_num_str}  {text.rstrip()}"
            
            lines.append(line_str)
        
        return '\n'.join(lines)
    
    def clear_cache(self):
        """Clear the source code cache."""
        self._source_cache.clear()
        self._bytecode_to_source.clear()
