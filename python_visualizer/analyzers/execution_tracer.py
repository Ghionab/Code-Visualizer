
import sys
import types
import dis
import time
from typing import List, Dict, Any, Optional, Callable, Iterator, Generator
from types import FrameType

from ..core.data_models import TraceEvent, FrameState
from ..core.config import Config


class ExecutionTrace:
    """Contains the complete execution trace results."""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.frame_states: List[FrameState] = []
        self.execution_time: float = 0.0
        self.total_events: int = 0
        self.visualization: str = ""


class ExecutionTracer:
    """Execution tracer using sys.settrace for runtime monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.trace_events: List[TraceEvent] = []
        self.current_frame: Optional[FrameType] = None
        self.start_time: float = 0.0
        self.is_tracing: bool = False
        self.step_mode: bool = False
        self.breakpoints: set = set()
        self.step_callback: Optional[Callable] = None
    
    def trace_execution(self, code: types.CodeType, globals_dict: Optional[Dict] = None) -> ExecutionTrace:
        """
        Trace the execution of compiled code.
        
        Args:
            code: Compiled code object to execute
            globals_dict: Optional globals dictionary
            
        Returns:
            ExecutionTrace containing all trace information
        """
        if globals_dict is None:
            globals_dict = {}
        
        # Reset trace state
        self.trace_events.clear()
        self.start_time = time.time()
        self.is_tracing = True
        
        # Set up tracing
        old_trace = sys.gettrace()
        sys.settrace(self.trace_function)
        
        try:
            # Execute the code
            exec(code, globals_dict)
            
        except Exception as e:
            # Record exception in trace
            if self.current_frame:
                exception_event = TraceEvent(
                    event_type='exception',
                    frame=self.current_frame,
                    instruction=None,
                    timestamp=time.time() - self.start_time,
                    stack_depth=len(self.trace_events),
                    arg=e
                )
                self.trace_events.append(exception_event)
        
        finally:
            # Restore previous trace function
            sys.settrace(old_trace)
            self.is_tracing = False
        
        # Create execution trace result
        trace = ExecutionTrace()
        trace.events = self.trace_events.copy()
        trace.execution_time = time.time() - self.start_time
        trace.total_events = len(self.trace_events)
        trace.visualization = self.visualize_trace(trace.events)
        
        return trace
    
    def trace_function(self, frame: FrameType, event: str, arg: Any) -> Callable:
        """
        Trace function for sys.settrace.
        
        Args:
            frame: Current execution frame
            event: Event type ('call', 'line', 'return', 'exception', 'opcode')
            arg: Event-specific argument
            
        Returns:
            Trace function to continue tracing
        """
        if not self.is_tracing:
            return self.trace_function
        
        # Check if we've hit the event limit
        if len(self.trace_events) >= self.config.tracing.max_trace_events:
            return None
        
        # Update current frame
        self.current_frame = frame
        
        # Create trace event
        trace_event = TraceEvent(
            event_type=event,
            frame=frame,
            instruction=self._get_current_instruction(frame),
            timestamp=time.time() - self.start_time,
            stack_depth=self._get_stack_depth(frame),
            arg=arg
        )
        
        # Record the event
        self.trace_events.append(trace_event)
        
        # Handle step mode and breakpoints
        if self.step_mode or self._should_break(frame):
            if self.step_callback:
                self.step_callback(trace_event)
        
        return self.trace_function
    
    def get_frame_state(self, frame: FrameType) -> FrameState:
        """
        Extract comprehensive frame state information.
        
        Args:
            frame: Frame to analyze
            
        Returns:
            FrameState containing complete frame information
        """
        return FrameState(
            locals=dict(frame.f_locals),
            globals=dict(frame.f_globals) if frame.f_globals else {},
            stack=self._get_evaluation_stack(frame),
            last_instruction=frame.f_lasti,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
            filename=frame.f_code.co_filename,
            code_name=frame.f_code.co_name
        )
    
    def visualize_trace(self, events: List[TraceEvent]) -> str:
        """
        Create a visual representation of the execution trace.
        
        Args:
            events: List of trace events
            
        Returns:
            Formatted string showing execution trace
        """
        if not events:
            return "No trace events to display"
        
        output = []
        output.append("=== EXECUTION TRACE ===\n")
        
        # Add summary statistics
        event_types = {}
        for event in events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        output.append("Trace Summary:")
        output.append(f"  Total events: {len(events)}")
        output.append(f"  Execution time: {events[-1].timestamp:.4f}s" if events else "  Execution time: 0s")
        output.append("  Event breakdown:")
        for event_type, count in sorted(event_types.items()):
            output.append(f"    {event_type}: {count}")
        output.append("")
        
        # Show detailed trace (limited for readability)
        max_events = min(50, len(events))
        output.append(f"Detailed Trace (showing first {max_events} events):")
        output.append("")
        
        for i, event in enumerate(events[:max_events]):
            timestamp_str = f"{event.timestamp:8.4f}s"
            depth_str = "  " * min(event.stack_depth, 10)  # Limit indentation
            
            # Format event based on type
            if event.event_type == 'call':
                func_name = event.frame.f_code.co_name
                output.append(f"{timestamp_str} {depth_str}→ CALL {func_name}() at line {event.frame.f_lineno}")
                
            elif event.event_type == 'line':
                line_num = event.frame.f_lineno
                func_name = event.frame.f_code.co_name
                output.append(f"{timestamp_str} {depth_str}  LINE {line_num} in {func_name}()")
                
            elif event.event_type == 'return':
                func_name = event.frame.f_code.co_name
                return_value = repr(event.arg) if event.arg is not None else "None"
                if len(return_value) > 30:
                    return_value = return_value[:27] + "..."
                output.append(f"{timestamp_str} {depth_str}← RETURN {func_name}() = {return_value}")
                
            elif event.event_type == 'exception':
                exc_type = type(event.arg).__name__ if event.arg else "Exception"
                output.append(f"{timestamp_str} {depth_str}! EXCEPTION {exc_type}: {event.arg}")
                
            else:
                output.append(f"{timestamp_str} {depth_str}  {event.event_type.upper()}")
        
        if len(events) > max_events:
            output.append(f"\n... and {len(events) - max_events} more events")
        
        return '\n'.join(output)
    
    def set_step_mode(self, enabled: bool, callback: Optional[Callable] = None):
        """Enable or disable step mode."""
        self.step_mode = enabled
        self.step_callback = callback
    
    def add_breakpoint(self, line_number: int):
        """Add a breakpoint at the specified line number."""
        self.breakpoints.add(line_number)
    
    def remove_breakpoint(self, line_number: int):
        """Remove a breakpoint at the specified line number."""
        self.breakpoints.discard(line_number)
    
    def clear_breakpoints(self):
        """Clear all breakpoints."""
        self.breakpoints.clear()
    
    def _get_current_instruction(self, frame: FrameType) -> Optional[dis.Instruction]:
        """Get the current bytecode instruction for the frame."""
        try:
            # Get instructions for the frame's code
            instructions = list(dis.get_instructions(frame.f_code))
            
            # Find instruction at current offset
            for instr in instructions:
                if instr.offset == frame.f_lasti:
                    return instr
                    
        except Exception:
            pass
        
        return None
    
    def _get_stack_depth(self, frame: FrameType) -> int:
        """Calculate the current stack depth."""
        depth = 0
        current = frame
        while current:
            depth += 1
            current = current.f_back
        return depth
    
    def _get_evaluation_stack(self, frame: FrameType) -> List[Any]:
        """
        Attempt to get the evaluation stack state.
        
        Note: Python doesn't expose the evaluation stack directly, so this
        is an approximation based on bytecode analysis and frame state.
        """
        stack_items = []
        
        try:
            # Get current instruction
            current_instr = self._get_current_instruction(frame)
            if not current_instr:
                return stack_items
            
            # Analyze recent instructions to estimate stack state
            instructions = list(dis.get_instructions(frame.f_code))
            current_offset = frame.f_lasti
            
            # Find instructions up to current position
            executed_instructions = []
            for instr in instructions:
                if instr.offset <= current_offset:
                    executed_instructions.append(instr)
                else:
                    break
            
            # Simulate stack effects (simplified)
            stack_depth = 0
            for instr in executed_instructions[-10:]:  # Look at last 10 instructions
                effect = self._get_instruction_stack_effect(instr)
                stack_depth += effect
                
                # Add approximated stack items based on instruction
                if instr.opname == 'LOAD_CONST':
                    if instr.argval is not None:
                        stack_items.append(instr.argval)
                elif instr.opname in ['LOAD_NAME', 'LOAD_GLOBAL', 'LOAD_FAST']:
                    if instr.argval:
                        # Try to get actual value from frame
                        try:
                            if instr.opname == 'LOAD_FAST':
                                value = frame.f_locals.get(instr.argval, f"<{instr.argval}>")
                            elif instr.opname == 'LOAD_GLOBAL':
                                value = frame.f_globals.get(instr.argval, f"<{instr.argval}>")
                            else:
                                value = frame.f_locals.get(instr.argval, 
                                       frame.f_globals.get(instr.argval, f"<{instr.argval}>"))
                            stack_items.append(value)
                        except:
                            stack_items.append(f"<{instr.argval}>")
                elif instr.opname == 'POP_TOP' and stack_items:
                    stack_items.pop()
            
            # Limit stack items for display
            return stack_items[-5:] if len(stack_items) > 5 else stack_items
            
        except Exception:
            # If stack analysis fails, return empty list
            return []
    
    def _get_instruction_stack_effect(self, instruction: dis.Instruction) -> int:
        """Get the stack effect of a bytecode instruction."""
        try:
            return dis.stack_effect(instruction.opcode, instruction.arg)
        except (ValueError, SystemError):
            return 0
    
    def visualize_stack_state(self, frame: FrameType) -> str:
        """
        Create a visualization of the current stack state.
        
        Args:
            frame: Current execution frame
            
        Returns:
            String representation of stack state
        """
        output = []
        output.append("=== STACK STATE ===")
        
        # Get frame state
        frame_state = self.get_frame_state(frame)
        
        # Show evaluation stack (approximated)
        stack = frame_state.stack
        if stack:
            output.append("\nEvaluation Stack (top to bottom):")
            for i, item in enumerate(reversed(stack)):
                item_repr = repr(item)
                if len(item_repr) > 40:
                    item_repr = item_repr[:37] + "..."
                output.append(f"  [{len(stack)-1-i}] {item_repr}")
        else:
            output.append("\nEvaluation Stack: <empty>")
        
        # Show local variables
        if frame_state.locals:
            output.append(f"\nLocal Variables ({len(frame_state.locals)}):")
            for name, value in sorted(frame_state.locals.items()):
                if not name.startswith('__'):  # Skip dunder variables
                    value_repr = repr(value)
                    if len(value_repr) > 30:
                        value_repr = value_repr[:27] + "..."
                    output.append(f"  {name} = {value_repr}")
        
        # Show current instruction
        current_instr = self._get_current_instruction(frame)
        if current_instr:
            output.append(f"\nCurrent Instruction:")
            output.append(f"  {current_instr.offset:4d} {current_instr.opname}")
            if current_instr.argval is not None:
                output.append(f"       arg: {current_instr.argval}")
        
        # Show frame info
        output.append(f"\nFrame Info:")
        output.append(f"  Function: {frame_state.function_name}")
        output.append(f"  Line: {frame_state.line_number}")
        output.append(f"  Last instruction: {frame_state.last_instruction}")
        
        return '\n'.join(output)
    
    def track_stack_changes(self, before_frame: FrameType, after_frame: FrameType) -> Dict[str, Any]:
        """
        Track changes in stack state between two frames.
        
        Args:
            before_frame: Frame state before instruction
            after_frame: Frame state after instruction
            
        Returns:
            Dictionary describing stack changes
        """
        before_state = self.get_frame_state(before_frame)
        after_state = self.get_frame_state(after_frame)
        
        changes = {
            'stack_size_change': len(after_state.stack) - len(before_state.stack),
            'locals_changed': [],
            'instruction_executed': None,
            'line_changed': after_state.line_number != before_state.line_number
        }
        
        # Check for local variable changes
        for name in set(before_state.locals.keys()) | set(after_state.locals.keys()):
            before_val = before_state.locals.get(name)
            after_val = after_state.locals.get(name)
            
            if before_val != after_val:
                changes['locals_changed'].append({
                    'name': name,
                    'before': before_val,
                    'after': after_val
                })
        
        # Get executed instruction
        current_instr = self._get_current_instruction(after_frame)
        if current_instr:
            changes['instruction_executed'] = {
                'opname': current_instr.opname,
                'arg': current_instr.arg,
                'argval': current_instr.argval,
                'offset': current_instr.offset
            }
        
        return changes
    
    def _should_break(self, frame: FrameType) -> bool:
        """Check if execution should break at this frame."""
        return frame.f_lineno in self.breakpoints
    
    def create_step_iterator(self, code: types.CodeType, 
                           globals_dict: Optional[Dict] = None) -> Generator[TraceEvent, None, None]:
        """
        Create an iterator for step-by-step execution.
        
        Args:
            code: Compiled code object
            globals_dict: Optional globals dictionary
            
        Yields:
            TraceEvent for each execution step
        """
        if globals_dict is None:
            globals_dict = {}
        
        # This is a simplified implementation
        # A full implementation would require more sophisticated control
        
        self.trace_events.clear()
        self.start_time = time.time()
        self.is_tracing = True
        
        # Set up step-by-step tracing
        step_events = []
        
        def step_trace_function(frame, event, arg):
            if not self.is_tracing:
                return None
            
            trace_event = TraceEvent(
                event_type=event,
                frame=frame,
                instruction=self._get_current_instruction(frame),
                timestamp=time.time() - self.start_time,
                stack_depth=self._get_stack_depth(frame),
                arg=arg
            )
            
            step_events.append(trace_event)
            return step_trace_function
        
        old_trace = sys.gettrace()
        sys.settrace(step_trace_function)
        
        try:
            exec(code, globals_dict)
        except Exception as e:
            # Handle exceptions
            pass
        finally:
            sys.settrace(old_trace)
            self.is_tracing = False
        
        # Yield events one by one
        for event in step_events:
            yield event


def create_tracer(config: Optional[Config] = None) -> ExecutionTracer:
    """Create an ExecutionTracer instance with optional configuration."""
    if config is None:
        from ..core.config import create_default_config
        config = create_default_config()
    
    return ExecutionTracer(config)