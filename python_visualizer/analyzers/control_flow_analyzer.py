"""
Control flow analyzer for conditional statements, loops, and exception handling.

This module provides analysis of control flow constructs including:
- Conditional statements (if/elif/else)
- Short-circuit evaluation (and/or operators)
- Loop iteration (for/while loops with iterator protocol)
- Exception propagation and try/except handling
"""

import dis
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from types import FrameType
from dataclasses import dataclass, field

from ..core.data_models import (
    ConditionalAnalysis, 
    LoopAnalysis, 
    ExceptionAnalysis,
    TraceEvent
)
from ..core.config import Config


@dataclass
class LoopState:
    """Tracks the state of an active loop."""
    loop_type: str  # 'for' or 'while'
    start_offset: int
    end_offset: int
    iterator_object: Any = None
    iteration_count: int = 0
    current_value: Any = None
    is_exhausted: bool = False
    loop_variable: Optional[str] = None


@dataclass
class ConditionalState:
    """Tracks the state of a conditional branch."""
    condition_offset: int
    branch_type: str  # 'if', 'elif', 'else', 'and', 'or'
    condition_value: Any = None
    branch_taken: bool = False
    short_circuit: bool = False


@dataclass
class ExceptionState:
    """Tracks exception handling state."""
    exception_type: type
    exception_value: Any
    handler_offset: Optional[int] = None
    propagation_frames: List[str] = field(default_factory=list)
    is_handled: bool = False


class ControlFlowAnalyzer:
    """
    Analyzes control flow constructs during Python code execution.
    
    This analyzer tracks:
    - Conditional branches and short-circuit evaluation
    - Loop iterations and iterator protocol mechanics
    - Exception propagation through call stack
    - Try/except handler selection
    """
    
    def __init__(self, config: Config):
        """
        Initialize the control flow analyzer.
        
        Args:
            config: Configuration object for analyzer settings
        """
        self.config = config
        self.active_loops: Dict[int, LoopState] = {}
        self.conditional_history: List[ConditionalState] = []
        self.exception_stack: List[ExceptionState] = []
        self.loop_counters: Dict[int, int] = {}
        
        # Opcodes related to control flow
        self.conditional_opcodes = {
            'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE',
            'JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP',
            'JUMP_IF_NOT_EXC_MATCH'
        }
        
        self.loop_opcodes = {
            'FOR_ITER', 'GET_ITER', 'JUMP_BACKWARD', 'JUMP_ABSOLUTE'
        }
        
        self.exception_opcodes = {
            'RAISE_VARARGS', 'RERAISE', 'POP_EXCEPT',
            'SETUP_FINALLY', 'SETUP_EXCEPT', 'JUMP_IF_NOT_EXC_MATCH'
        }
    
    def analyze_conditional(self, frame: FrameType, 
                          instruction: dis.Instruction) -> ConditionalAnalysis:
        """
        Analyze conditional branch decisions.
        
        This method examines conditional statements and determines:
        - The condition value being tested
        - Which branch will be taken
        - Whether short-circuit evaluation occurred
        
        Args:
            frame: Current execution frame
            instruction: Conditional jump instruction
            
        Returns:
            ConditionalAnalysis object with branch decision details
        """
        opname = instruction.opname
        
        # Try to get the condition value from the stack
        condition_value = self._get_condition_value(frame, instruction)
        
        # Determine branch taken
        branch_taken = self._determine_branch(opname, condition_value, instruction)
        
        # Check for short-circuit evaluation
        short_circuit = self._detect_short_circuit(frame, instruction)
        
        # Generate explanation
        explanation = self._explain_conditional(
            opname, condition_value, branch_taken, short_circuit, instruction
        )
        
        # Get source line if available
        source_line = self._get_source_line(frame)
        
        # Record conditional state
        cond_state = ConditionalState(
            condition_offset=instruction.offset,
            branch_type=self._classify_branch_type(opname),
            condition_value=condition_value,
            branch_taken=branch_taken,
            short_circuit=short_circuit
        )
        self.conditional_history.append(cond_state)
        
        return ConditionalAnalysis(
            condition_value=condition_value,
            branch_taken='true' if branch_taken else 'false',
            short_circuit=short_circuit,
            explanation=explanation,
            source_line=source_line
        )
    
    def analyze_loop(self, frame: FrameType, 
                    instruction: dis.Instruction) -> LoopAnalysis:
        """
        Analyze loop iteration mechanics.
        
        This method tracks:
        - Iterator object creation and protocol
        - Current iteration value
        - Loop termination (StopIteration)
        - Iteration count
        
        Args:
            frame: Current execution frame
            instruction: Loop-related instruction
            
        Returns:
            LoopAnalysis object with iteration details
        """
        opname = instruction.opname
        offset = instruction.offset
        
        # Handle different loop opcodes
        if opname == 'GET_ITER':
            return self._analyze_get_iter(frame, instruction)
        elif opname == 'FOR_ITER':
            return self._analyze_for_iter(frame, instruction)
        elif opname in ('JUMP_BACKWARD', 'JUMP_ABSOLUTE'):
            return self._analyze_loop_jump(frame, instruction)
        else:
            # Generic loop analysis
            return self._analyze_generic_loop(frame, instruction)
    
    def analyze_exception(self, frame: FrameType, 
                         exception: Exception) -> ExceptionAnalysis:
        """
        Analyze exception propagation and handling.
        
        This method tracks:
        - Exception type and value
        - Propagation through call stack
        - Try/except handler selection
        - Exception re-raising
        
        Args:
            frame: Current execution frame
            exception: Exception object being raised
            
        Returns:
            ExceptionAnalysis object with exception handling details
        """
        exception_type = type(exception)
        exception_value = exception
        
        # Build traceback information
        traceback_info = self._build_traceback_info(frame, exception)
        
        # Check if handler exists in current frame
        handler_found, handler_offset = self._find_exception_handler(
            frame, exception_type
        )
        
        # Build propagation path
        propagation_path = self._build_propagation_path(frame)
        
        # Generate explanation
        explanation = self._explain_exception(
            exception_type, exception_value, handler_found, propagation_path
        )
        
        # Record exception state
        exc_state = ExceptionState(
            exception_type=exception_type,
            exception_value=exception_value,
            handler_offset=handler_offset,
            propagation_frames=propagation_path,
            is_handled=handler_found
        )
        self.exception_stack.append(exc_state)
        
        return ExceptionAnalysis(
            exception_type=exception_type,
            exception_value=exception_value,
            traceback_info=traceback_info,
            handler_found=handler_found,
            propagation_path=propagation_path,
            explanation=explanation
        )
    
    def detect_short_circuit_evaluation(self, frame: FrameType,
                                       instruction: dis.Instruction) -> bool:
        """
        Detect if short-circuit evaluation is occurring.
        
        Short-circuit evaluation happens with 'and' and 'or' operators
        when the result can be determined without evaluating all operands.
        
        Args:
            frame: Current execution frame
            instruction: Current instruction
            
        Returns:
            True if short-circuit evaluation is detected
        """
        return self._detect_short_circuit(frame, instruction)
    
    def get_loop_state(self, offset: int) -> Optional[LoopState]:
        """
        Get the state of a loop at a specific bytecode offset.
        
        Args:
            offset: Bytecode offset
            
        Returns:
            LoopState if a loop is active at that offset, None otherwise
        """
        return self.active_loops.get(offset)
    
    def get_conditional_history(self) -> List[ConditionalState]:
        """
        Get the history of all conditional branches analyzed.
        
        Returns:
            List of ConditionalState objects
        """
        return self.conditional_history.copy()
    
    def get_exception_stack(self) -> List[ExceptionState]:
        """
        Get the current exception handling stack.
        
        Returns:
            List of ExceptionState objects
        """
        return self.exception_stack.copy()
    
    def reset(self):
        """Reset analyzer state for new analysis."""
        self.active_loops.clear()
        self.conditional_history.clear()
        self.exception_stack.clear()
        self.loop_counters.clear()
    
    # Private helper methods
    
    def _get_condition_value(self, frame: FrameType, 
                            instruction: dis.Instruction) -> Any:
        """Extract the condition value being tested."""
        try:
            # Try to get value from local variables based on instruction
            if instruction.argval is not None:
                # Check locals first
                if instruction.argval in frame.f_locals:
                    return frame.f_locals[instruction.argval]
                # Check globals
                if instruction.argval in frame.f_globals:
                    return frame.f_globals[instruction.argval]
            
            # For comparison operations, try to infer from recent instructions
            # This is a simplified approach - real implementation would need
            # to track the evaluation stack more carefully
            return None
            
        except Exception:
            return None
    
    def _determine_branch(self, opname: str, condition_value: Any,
                         instruction: dis.Instruction) -> bool:
        """Determine which branch will be taken."""
        if opname == 'POP_JUMP_IF_FALSE':
            return not bool(condition_value) if condition_value is not None else False
        elif opname == 'POP_JUMP_IF_TRUE':
            return bool(condition_value) if condition_value is not None else True
        elif opname == 'JUMP_IF_FALSE_OR_POP':
            return not bool(condition_value) if condition_value is not None else False
        elif opname == 'JUMP_IF_TRUE_OR_POP':
            return bool(condition_value) if condition_value is not None else True
        else:
            return False
    
    def _detect_short_circuit(self, frame: FrameType,
                             instruction: dis.Instruction) -> bool:
        """Detect short-circuit evaluation in boolean expressions."""
        opname = instruction.opname
        
        # Short-circuit opcodes
        if opname in ('JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP'):
            return True
        
        # Check if this is part of an 'and' or 'or' expression
        # by examining nearby instructions
        try:
            instructions = list(dis.get_instructions(frame.f_code))
            current_idx = None
            
            for idx, instr in enumerate(instructions):
                if instr.offset == instruction.offset:
                    current_idx = idx
                    break
            
            if current_idx is not None and current_idx > 0:
                prev_instr = instructions[current_idx - 1]
                # Check for patterns indicating short-circuit
                if prev_instr.opname in ('COMPARE_OP', 'LOAD_CONST', 'LOAD_FAST'):
                    if opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'):
                        # Look ahead to see if there's another conditional
                        if current_idx < len(instructions) - 1:
                            next_instr = instructions[current_idx + 1]
                            if next_instr.opname in self.conditional_opcodes:
                                return True
        except Exception:
            pass
        
        return False
    
    def _classify_branch_type(self, opname: str) -> str:
        """Classify the type of conditional branch."""
        if opname in ('JUMP_IF_FALSE_OR_POP', 'JUMP_IF_TRUE_OR_POP'):
            return 'short_circuit'
        elif opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'):
            return 'if'
        else:
            return 'conditional'
    
    def _explain_conditional(self, opname: str, condition_value: Any,
                           branch_taken: bool, short_circuit: bool,
                           instruction: dis.Instruction) -> str:
        """Generate human-readable explanation of conditional."""
        parts = []
        
        if short_circuit:
            if opname == 'JUMP_IF_FALSE_OR_POP':
                parts.append("Short-circuit 'and' evaluation:")
                if not branch_taken:
                    parts.append("First operand is False, skipping second operand")
                else:
                    parts.append("First operand is True, evaluating second operand")
            elif opname == 'JUMP_IF_TRUE_OR_POP':
                parts.append("Short-circuit 'or' evaluation:")
                if branch_taken:
                    parts.append("First operand is True, skipping second operand")
                else:
                    parts.append("First operand is False, evaluating second operand")
        else:
            parts.append(f"Conditional branch ({opname}):")
            if condition_value is not None:
                parts.append(f"Condition evaluates to {bool(condition_value)}")
            
            if branch_taken:
                parts.append(f"Taking branch to offset {instruction.argval}")
            else:
                parts.append("Continuing to next instruction")
        
        return " ".join(parts)
    
    def _analyze_get_iter(self, frame: FrameType,
                         instruction: dis.Instruction) -> LoopAnalysis:
        """Analyze GET_ITER instruction (iterator creation)."""
        # Try to get the iterable object
        iterator_object = None
        try:
            # The iterable should be on top of stack (we can't access it directly)
            # but we can infer from locals
            for name, value in frame.f_locals.items():
                if hasattr(value, '__iter__'):
                    iterator_object = value
                    break
        except Exception:
            pass
        
        # Create loop state
        loop_state = LoopState(
            loop_type='for',
            start_offset=instruction.offset,
            end_offset=-1,  # Will be determined later
            iterator_object=iterator_object,
            iteration_count=0
        )
        self.active_loops[instruction.offset] = loop_state
        
        explanation = (
            f"Creating iterator for 'for' loop. "
            f"Iterator protocol: calling __iter__() on iterable object"
        )
        
        if iterator_object is not None:
            explanation += f" (iterating over {type(iterator_object).__name__})"
        
        return LoopAnalysis(
            iterator_object=iterator_object,
            current_value=None,
            iteration_count=0,
            loop_type='for',
            explanation=explanation,
            is_exhausted=False
        )
    
    def _analyze_for_iter(self, frame: FrameType,
                         instruction: dis.Instruction) -> LoopAnalysis:
        """Analyze FOR_ITER instruction (loop iteration)."""
        # Find associated loop state
        loop_state = None
        for offset, state in self.active_loops.items():
            if state.loop_type == 'for' and not state.is_exhausted:
                loop_state = state
                break
        
        if loop_state is None:
            # Create new loop state if not found
            loop_state = LoopState(
                loop_type='for',
                start_offset=instruction.offset,
                end_offset=instruction.argval if instruction.argval else -1,
                iteration_count=0
            )
            self.active_loops[instruction.offset] = loop_state
        
        # Increment iteration count
        loop_state.iteration_count += 1
        self.loop_counters[instruction.offset] = loop_state.iteration_count
        
        # Try to get current value from locals
        current_value = None
        try:
            # Look for recently assigned loop variable
            for name, value in frame.f_locals.items():
                if not name.startswith('_'):
                    current_value = value
                    loop_state.current_value = value
                    loop_state.loop_variable = name
                    break
        except Exception:
            pass
        
        # Check if loop is exhausted (StopIteration)
        # This would be indicated by a jump to the end offset
        is_exhausted = False
        
        explanation = (
            f"Loop iteration {loop_state.iteration_count}: "
            f"Calling __next__() on iterator"
        )
        
        if current_value is not None:
            value_repr = repr(current_value)
            if len(value_repr) > 30:
                value_repr = value_repr[:27] + "..."
            explanation += f", got value: {value_repr}"
        
        if is_exhausted:
            explanation += ". StopIteration raised, exiting loop"
            loop_state.is_exhausted = True
        
        return LoopAnalysis(
            iterator_object=loop_state.iterator_object,
            current_value=current_value,
            iteration_count=loop_state.iteration_count,
            loop_type='for',
            explanation=explanation,
            is_exhausted=is_exhausted
        )
    
    def _analyze_loop_jump(self, frame: FrameType,
                          instruction: dis.Instruction) -> LoopAnalysis:
        """Analyze loop jump instructions (JUMP_BACKWARD, JUMP_ABSOLUTE)."""
        # This indicates a loop continuation or while loop
        loop_type = 'while'  # Assume while loop for backward jumps
        
        # Check if this is part of an existing loop
        target_offset = instruction.argval if instruction.argval else 0
        loop_state = self.active_loops.get(target_offset)
        
        if loop_state is None:
            # Create new loop state for while loop
            loop_state = LoopState(
                loop_type=loop_type,
                start_offset=target_offset,
                end_offset=instruction.offset,
                iteration_count=1
            )
            self.active_loops[target_offset] = loop_state
        else:
            loop_state.iteration_count += 1
        
        explanation = (
            f"Loop continuation: jumping back to offset {target_offset} "
            f"(iteration {loop_state.iteration_count})"
        )
        
        return LoopAnalysis(
            iterator_object=None,
            current_value=None,
            iteration_count=loop_state.iteration_count,
            loop_type=loop_type,
            explanation=explanation,
            is_exhausted=False
        )
    
    def _analyze_generic_loop(self, frame: FrameType,
                             instruction: dis.Instruction) -> LoopAnalysis:
        """Generic loop analysis for unrecognized loop patterns."""
        return LoopAnalysis(
            iterator_object=None,
            current_value=None,
            iteration_count=0,
            loop_type='unknown',
            explanation=f"Loop-related instruction: {instruction.opname}",
            is_exhausted=False
        )
    
    def _build_traceback_info(self, frame: FrameType,
                             exception: Exception) -> List[Dict[str, Any]]:
        """Build traceback information from frame and exception."""
        traceback_info = []
        
        current_frame = frame
        while current_frame is not None:
            frame_info = {
                'filename': current_frame.f_code.co_filename,
                'function': current_frame.f_code.co_name,
                'line_number': current_frame.f_lineno,
                'locals': dict(current_frame.f_locals)
            }
            traceback_info.append(frame_info)
            current_frame = current_frame.f_back
        
        return traceback_info
    
    def _find_exception_handler(self, frame: FrameType,
                               exception_type: type) -> Tuple[bool, Optional[int]]:
        """Find exception handler in current frame."""
        try:
            # Examine bytecode for exception handling blocks
            instructions = list(dis.get_instructions(frame.f_code))
            
            # Look for exception handling opcodes
            for instr in instructions:
                if instr.opname in ('SETUP_FINALLY', 'SETUP_EXCEPT', 'POP_EXCEPT'):
                    # Found exception handler
                    return True, instr.offset
            
            return False, None
            
        except Exception:
            return False, None
    
    def _build_propagation_path(self, frame: FrameType) -> List[str]:
        """Build exception propagation path through call stack."""
        path = []
        
        current_frame = frame
        while current_frame is not None:
            func_name = current_frame.f_code.co_name
            line_num = current_frame.f_lineno
            path.append(f"{func_name}() at line {line_num}")
            current_frame = current_frame.f_back
        
        return path
    
    def _explain_exception(self, exception_type: type, exception_value: Any,
                          handler_found: bool, propagation_path: List[str]) -> str:
        """Generate human-readable explanation of exception handling."""
        parts = []
        
        exc_name = exception_type.__name__
        parts.append(f"Exception raised: {exc_name}")
        
        if exception_value:
            exc_str = str(exception_value)
            if len(exc_str) > 50:
                exc_str = exc_str[:47] + "..."
            parts.append(f"with message: {exc_str}")
        
        if handler_found:
            parts.append("Exception handler found in current frame")
        else:
            parts.append("No handler found, propagating to caller")
        
        if propagation_path:
            parts.append(f"Propagation path: {' -> '.join(propagation_path[:3])}")
            if len(propagation_path) > 3:
                parts.append(f"... and {len(propagation_path) - 3} more frames")
        
        return ". ".join(parts)
    
    def _get_source_line(self, frame: FrameType) -> Optional[str]:
        """Get the source code line for the current frame."""
        try:
            import linecache
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno
            line = linecache.getline(filename, line_number).strip()
            return line if line else None
        except Exception:
            return None
    
    def visualize_control_flow(self, frame: FrameType) -> str:
        """
        Create a visualization of current control flow state.
        
        Args:
            frame: Current execution frame
            
        Returns:
            Formatted string showing control flow state
        """
        output = []
        output.append("=== CONTROL FLOW STATE ===")
        output.append("")
        
        # Show active loops
        if self.active_loops:
            output.append("Active Loops:")
            for offset, loop_state in self.active_loops.items():
                if not loop_state.is_exhausted:
                    output.append(f"  {loop_state.loop_type.upper()} loop at offset {offset}")
                    output.append(f"    Iterations: {loop_state.iteration_count}")
                    if loop_state.loop_variable:
                        output.append(f"    Variable: {loop_state.loop_variable}")
                    if loop_state.current_value is not None:
                        value_repr = repr(loop_state.current_value)
                        if len(value_repr) > 40:
                            value_repr = value_repr[:37] + "..."
                        output.append(f"    Current value: {value_repr}")
            output.append("")
        
        # Show recent conditionals
        if self.conditional_history:
            output.append("Recent Conditionals (last 5):")
            for cond in self.conditional_history[-5:]:
                output.append(f"  {cond.branch_type.upper()} at offset {cond.condition_offset}")
                output.append(f"    Branch taken: {cond.branch_taken}")
                if cond.short_circuit:
                    output.append(f"    Short-circuit: Yes")
            output.append("")
        
        # Show exception stack
        if self.exception_stack:
            output.append("Exception Stack:")
            for exc in self.exception_stack:
                output.append(f"  {exc.exception_type.__name__}: {exc.exception_value}")
                output.append(f"    Handled: {exc.is_handled}")
                if exc.propagation_frames:
                    output.append(f"    Frames: {len(exc.propagation_frames)}")
            output.append("")
        
        if not self.active_loops and not self.conditional_history and not self.exception_stack:
            output.append("No active control flow constructs")
        
        return '\n'.join(output)


def create_control_flow_analyzer(config: Optional[Config] = None) -> ControlFlowAnalyzer:
    """
    Create a ControlFlowAnalyzer instance with optional configuration.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured ControlFlowAnalyzer instance
    """
    if config is None:
        from ..core.config import create_default_config
        config = create_default_config()
    
    return ControlFlowAnalyzer(config)
