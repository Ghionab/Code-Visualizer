
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from types import FrameType
import dis
import tracemalloc


@dataclass
class TraceEvent:
    """Represents a single execution event captured during tracing."""
    event_type: str  # 'call', 'line', 'return', 'exception', 'opcode'
    frame: FrameType
    instruction: Optional[dis.Instruction]
    timestamp: float
    stack_depth: int
    arg: Any = None  # Additional event-specific data


@dataclass
class FrameState:
    """Captures the complete state of a Python frame at a point in time."""
    locals: Dict[str, Any]
    globals: Dict[str, Any]
    stack: List[Any]
    last_instruction: int
    line_number: int
    function_name: str
    filename: str
    code_name: str


@dataclass
class AllocationInfo:
    """Information about a memory allocation event."""
    object_id: int
    object_type: type
    size: int
    traceback: Optional[tracemalloc.Traceback]
    timestamp: float
    reference_count: int = 0


@dataclass
class MemoryLayout:
    """Represents the memory layout of a Python object."""
    object_id: int
    type_name: str
    size: int
    reference_count: int
    fields: Dict[str, Any]
    memory_address: int
    internal_structure: Optional[Dict[str, Any]] = None


@dataclass
class ConditionalAnalysis:
    """Analysis results for conditional statements and branches."""
    condition_value: Any
    branch_taken: str  # 'true', 'false', 'short_circuit'
    short_circuit: bool
    explanation: str
    source_line: Optional[str] = None


@dataclass
class LoopAnalysis:
    """Analysis results for loop constructs and iteration."""
    iterator_object: Any
    current_value: Any
    iteration_count: int
    loop_type: str  # 'for', 'while'
    explanation: str
    is_exhausted: bool = False


@dataclass
class CompilationStage:
    """Represents a stage in the compilation pipeline."""
    stage_name: str  # 'tokenize', 'parse', 'compile'
    input_data: Any
    output_data: Any
    explanation: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedInstruction:
    """Enhanced bytecode instruction with additional analysis information."""
    instruction: dis.Instruction
    stack_effect: int
    explanation: str
    source_line: Optional[str]
    related_objects: List[Any]
    execution_count: int = 0


@dataclass
class ExceptionAnalysis:
    """Analysis of exception handling and propagation."""
    exception_type: type
    exception_value: Any
    traceback_info: List[Dict[str, Any]]
    handler_found: bool
    propagation_path: List[str]
    explanation: str


@dataclass
class IOOperation:
    """Information about I/O operations during execution."""
    operation_type: str  # 'read', 'write', 'open', 'close'
    target: str  # file path, stdout, stderr, etc.
    data: Any
    result: Any
    system_call: Optional[str]
    timestamp: float


@dataclass
class FunctionCallInfo:
    """Information about function calls and argument binding."""
    function_name: str
    arguments: Dict[str, Any]
    local_scope: Dict[str, Any]
    closure_vars: Dict[str, Any]
    return_value: Any = None
    call_depth: int = 0


# Type aliases for commonly used collections
TraceEventList = List[TraceEvent]
AllocationMap = Dict[int, AllocationInfo]
FrameStack = List[FrameState]
InstructionSequence = List[EnhancedInstruction]