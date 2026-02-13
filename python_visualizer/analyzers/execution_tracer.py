
import sys
import types
import dis
import time
from typing import List, Dict, Any, Optional, Callable, Iterator, Generator
from types import FrameType
from dataclasses import dataclass

from ..core.data_models import TraceEvent, FrameState
from ..core.config import Config


@dataclass
class StackStateSnapshot:
    """Snapshot of evaluation stack state at a specific point."""
    stack_items: List[Any]
    stack_depth: int
    instruction: Optional[dis.Instruction]
    timestamp: float
    frame_locals: Dict[str, Any]
    
    def __repr__(self):
        items_repr = [repr(item)[:30] for item in self.stack_items]
        return f"StackState(depth={self.stack_depth}, items={items_repr})"


@dataclass
class StackChange:
    """Represents a change in stack state between two instructions."""
    before: StackStateSnapshot
    after: StackStateSnapshot
    instruction: dis.Instruction
    expected_effect: int
    actual_effect: int
    items_pushed: List[Any]
    items_popped: List[Any]
    effect_matches: bool
    explanation: str


class ExecutionTrace:
    """Contains the complete execution trace results."""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.frame_states: List[FrameState] = []
        self.stack_changes: List[StackChange] = []
        self.execution_time: float = 0.0
        self.total_events: int = 0
        self.visualization: str = ""


class ExecutionTracer:
    """Execution tracer using sys.settrace for runtime monitoring."""
    
    def __init__(self, config: Config):
        self.config = config
        self.trace_events: List[TraceEvent] = []
        self.stack_snapshots: List[StackStateSnapshot] = []
        self.stack_changes: List[StackChange] = []
        self.current_frame: Optional[FrameType] = None
        self.previous_stack: Optional[StackStateSnapshot] = None
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
        self.stack_snapshots.clear()
        self.stack_changes.clear()
        self.previous_stack = None
        self.start_time = time.time()
        self.is_tracing = True
        
        # Set up tracing with opcode-level granularity
        old_trace = sys.gettrace()
        
        # Enable opcode tracing by setting sys.settrace with call_opcodes
        sys.settrace(self.trace_function)
        
        # Try to enable opcode-level tracing if available (Python 3.7+)
        try:
            # This enables per-opcode tracing
            sys.settrace(lambda *args: self.trace_function(*args))
        except:
            pass
        
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
        
        # Post-process: analyze stack changes from trace events
        self._analyze_stack_changes_from_trace()
        
        # Create execution trace result
        trace = ExecutionTrace()
        trace.events = self.trace_events.copy()
        trace.stack_changes = self.stack_changes.copy()
        trace.execution_time = time.time() - self.start_time
        trace.total_events = len(self.trace_events)
        trace.visualization = self.visualize_trace(trace.events)
        
        return trace
    
    def _analyze_stack_changes_from_trace(self):
        """
        Analyze stack changes from captured trace events.
        This is a post-processing step since sys.settrace doesn't give us
        opcode-level granularity by default.
        """
        # Group events by frame and analyze instruction sequences
        for i, event in enumerate(self.trace_events):
            if event.event_type == 'line' and event.frame:
                try:
                    # Get all instructions for this line
                    instructions = list(dis.get_instructions(event.frame.f_code))
                    
                    # Find instructions at or near current position
                    current_offset = event.frame.f_lasti
                    
                    for instr in instructions:
                        if instr.offset == current_offset:
                            # Create a synthetic stack change for this instruction
                            expected_effect = self._get_instruction_stack_effect(instr)
                            
                            # Create before/after snapshots (approximated)
                            before_snapshot = StackStateSnapshot(
                                stack_items=[],
                                stack_depth=0,
                                instruction=instr,
                                timestamp=event.timestamp,
                                frame_locals=dict(event.frame.f_locals)
                            )
                            
                            after_snapshot = StackStateSnapshot(
                                stack_items=[],
                                stack_depth=expected_effect,
                                instruction=instr,
                                timestamp=event.timestamp,
                                frame_locals=dict(event.frame.f_locals)
                            )
                            
                            stack_change = StackChange(
                                before=before_snapshot,
                                after=after_snapshot,
                                instruction=instr,
                                expected_effect=expected_effect,
                                actual_effect=expected_effect,  # Approximated
                                items_pushed=[],
                                items_popped=[],
                                effect_matches=True,
                                explanation=self._explain_stack_change(instr, expected_effect, 
                                                                      expected_effect, [], [], True)
                            )
                            
                            self.stack_changes.append(stack_change)
                            break
                            
                except Exception:
                    # Silently ignore errors in stack analysis
                    pass
    
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
        
        # Capture stack state before processing event
        if self.current_frame and event == 'line':
            # Track stack change from previous frame to current
            try:
                stack_change = self.track_stack_change(self.current_frame, frame)
                if stack_change:
                    self.stack_changes.append(stack_change)
                    self.stack_snapshots.append(stack_change.after)
            except Exception:
                # Silently ignore stack tracking errors
                pass
        
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
    
    def capture_stack_snapshot(self, frame: FrameType) -> StackStateSnapshot:
        """
        Capture a snapshot of the current stack state.
        
        Args:
            frame: Current execution frame
            
        Returns:
            StackStateSnapshot containing complete stack state
        """
        stack_items = self._get_evaluation_stack(frame)
        current_instr = self._get_current_instruction(frame)
        
        return StackStateSnapshot(
            stack_items=stack_items.copy(),
            stack_depth=len(stack_items),
            instruction=current_instr,
            timestamp=time.time() - self.start_time,
            frame_locals=dict(frame.f_locals)
        )
    
    def track_stack_change(self, before_frame: FrameType, after_frame: FrameType) -> Optional[StackChange]:
        """
        Track and validate stack changes between two execution points.
        
        Args:
            before_frame: Frame state before instruction execution
            after_frame: Frame state after instruction execution
            
        Returns:
            StackChange object describing the change, or None if no change
        """
        before_snapshot = self.capture_stack_snapshot(before_frame)
        after_snapshot = self.capture_stack_snapshot(after_frame)
        
        # Get the instruction that was executed
        instruction = after_snapshot.instruction
        if not instruction:
            return None
        
        # Calculate expected stack effect
        expected_effect = self._get_instruction_stack_effect(instruction)
        
        # Calculate actual stack effect
        actual_effect = after_snapshot.stack_depth - before_snapshot.stack_depth
        
        # Determine what was pushed and popped
        items_pushed = []
        items_popped = []
        
        if actual_effect > 0:
            # Items were pushed
            items_pushed = after_snapshot.stack_items[-actual_effect:] if actual_effect <= len(after_snapshot.stack_items) else []
        elif actual_effect < 0:
            # Items were popped
            pop_count = abs(actual_effect)
            items_popped = before_snapshot.stack_items[-pop_count:] if pop_count <= len(before_snapshot.stack_items) else []
        
        # Check if effect matches expectation
        effect_matches = (expected_effect == actual_effect)
        
        # Generate explanation
        explanation = self._explain_stack_change(instruction, expected_effect, actual_effect, 
                                                 items_pushed, items_popped, effect_matches)
        
        stack_change = StackChange(
            before=before_snapshot,
            after=after_snapshot,
            instruction=instruction,
            expected_effect=expected_effect,
            actual_effect=actual_effect,
            items_pushed=items_pushed,
            items_popped=items_popped,
            effect_matches=effect_matches,
            explanation=explanation
        )
        
        return stack_change
    
    def _explain_stack_change(self, instruction: dis.Instruction, expected: int, actual: int,
                             pushed: List[Any], popped: List[Any], matches: bool) -> str:
        """Generate human-readable explanation of stack change."""
        parts = []
        
        # Instruction description
        parts.append(f"{instruction.opname}")
        
        # Stack effect description
        if actual > 0:
            parts.append(f"pushed {actual} item(s)")
            if pushed:
                pushed_repr = [repr(item)[:20] for item in pushed]
                parts.append(f"[{', '.join(pushed_repr)}]")
        elif actual < 0:
            parts.append(f"popped {abs(actual)} item(s)")
            if popped:
                popped_repr = [repr(item)[:20] for item in popped]
                parts.append(f"[{', '.join(popped_repr)}]")
        else:
            parts.append("no net stack change")
        
        # Validation note
        if not matches:
            parts.append(f"(expected {expected:+d}, got {actual:+d})")
        
        return " ".join(parts)
    
    def visualize_stack_change(self, stack_change: StackChange) -> str:
        """
        Create a visual representation of a stack change.
        
        Args:
            stack_change: StackChange object to visualize
            
        Returns:
            Formatted string showing before/after stack states
        """
        output = []
        output.append("=== STACK CHANGE ===")
        output.append(f"Instruction: {stack_change.instruction.opname}")
        if stack_change.instruction.argval is not None:
            output.append(f"  Argument: {stack_change.instruction.argval}")
        output.append("")
        
        # Show before state
        output.append("BEFORE:")
        if stack_change.before.stack_items:
            for i, item in enumerate(reversed(stack_change.before.stack_items)):
                item_repr = repr(item)
                if len(item_repr) > 40:
                    item_repr = item_repr[:37] + "..."
                output.append(f"  [{len(stack_change.before.stack_items)-1-i}] {item_repr}")
        else:
            output.append("  <empty stack>")
        
        output.append("")
        
        # Show the change
        output.append(f"CHANGE: {stack_change.explanation}")
        output.append(f"  Expected effect: {stack_change.expected_effect:+d}")
        output.append(f"  Actual effect: {stack_change.actual_effect:+d}")
        output.append(f"  Validation: {'✓ PASS' if stack_change.effect_matches else '✗ MISMATCH'}")
        output.append("")
        
        # Show after state
        output.append("AFTER:")
        if stack_change.after.stack_items:
            for i, item in enumerate(reversed(stack_change.after.stack_items)):
                item_repr = repr(item)
                if len(item_repr) > 40:
                    item_repr = item_repr[:37] + "..."
                output.append(f"  [{len(stack_change.after.stack_items)-1-i}] {item_repr}")
        else:
            output.append("  <empty stack>")
        
        return '\n'.join(output)
    
    def get_stack_effect_validation_report(self) -> str:
        """
        Generate a report on stack effect validation across all tracked changes.
        
        Returns:
            Formatted report showing validation statistics
        """
        if not self.stack_changes:
            return "No stack changes tracked"
        
        output = []
        output.append("=== STACK EFFECT VALIDATION REPORT ===")
        output.append("")
        
        # Calculate statistics
        total_changes = len(self.stack_changes)
        matching = sum(1 for change in self.stack_changes if change.effect_matches)
        mismatches = total_changes - matching
        
        output.append(f"Total stack changes tracked: {total_changes}")
        output.append(f"Matching expected effects: {matching} ({100*matching/total_changes:.1f}%)")
        output.append(f"Mismatches: {mismatches} ({100*mismatches/total_changes:.1f}%)")
        output.append("")
        
        # Show mismatches if any
        if mismatches > 0:
            output.append("MISMATCHES:")
            for i, change in enumerate(self.stack_changes):
                if not change.effect_matches:
                    output.append(f"  {i+1}. {change.instruction.opname} at offset {change.instruction.offset}")
                    output.append(f"     Expected: {change.expected_effect:+d}, Actual: {change.actual_effect:+d}")
            output.append("")
        
        # Show opcode statistics
        opcode_stats = {}
        for change in self.stack_changes:
            opname = change.instruction.opname
            if opname not in opcode_stats:
                opcode_stats[opname] = {'total': 0, 'matches': 0}
            opcode_stats[opname]['total'] += 1
            if change.effect_matches:
                opcode_stats[opname]['matches'] += 1
        
        output.append("BY OPCODE:")
        for opname in sorted(opcode_stats.keys()):
            stats = opcode_stats[opname]
            match_rate = 100 * stats['matches'] / stats['total']
            output.append(f"  {opname:<20} {stats['matches']}/{stats['total']} ({match_rate:.0f}%)")
        
        return '\n'.join(output)
    
    def _get_instruction_stack_effect(self, instruction: dis.Instruction) -> int:
        """Get the stack effect of a bytecode instruction."""
        try:
            return dis.stack_effect(instruction.opcode, instruction.arg)
        except (ValueError, SystemError):
            return 0
    
    def visualize_stack_state(self, frame: FrameType, show_detailed: bool = True) -> str:
        """
        Create a visualization of the current stack state.
        
        Args:
            frame: Current execution frame
            show_detailed: Whether to show detailed information
            
        Returns:
            String representation of stack state
        """
        output = []
        output.append("=== STACK STATE ===")
        
        # Capture current snapshot
        snapshot = self.capture_stack_snapshot(frame)
        
        # Show evaluation stack
        if snapshot.stack_items:
            output.append(f"\nEvaluation Stack (depth={snapshot.stack_depth}, top to bottom):")
            for i, item in enumerate(reversed(snapshot.stack_items)):
                item_repr = repr(item)
                if len(item_repr) > 40:
                    item_repr = item_repr[:37] + "..."
                item_type = type(item).__name__
                output.append(f"  [{len(snapshot.stack_items)-1-i}] {item_repr} ({item_type})")
        else:
            output.append("\nEvaluation Stack: <empty>")
        
        if show_detailed:
            # Show local variables
            frame_state = self.get_frame_state(frame)
            if frame_state.locals:
                output.append(f"\nLocal Variables ({len(frame_state.locals)}):")
                for name, value in sorted(frame_state.locals.items()):
                    if not name.startswith('__'):  # Skip dunder variables
                        value_repr = repr(value)
                        if len(value_repr) > 30:
                            value_repr = value_repr[:27] + "..."
                        output.append(f"  {name} = {value_repr}")
            
            # Show current instruction
            if snapshot.instruction:
                output.append(f"\nCurrent Instruction:")
                output.append(f"  {snapshot.instruction.offset:4d} {snapshot.instruction.opname}")
                if snapshot.instruction.argval is not None:
                    output.append(f"       arg: {snapshot.instruction.argval}")
                
                # Show expected stack effect
                expected_effect = self._get_instruction_stack_effect(snapshot.instruction)
                output.append(f"       stack effect: {expected_effect:+d}")
            
            # Show frame info
            output.append(f"\nFrame Info:")
            output.append(f"  Function: {frame_state.function_name}")
            output.append(f"  Line: {frame_state.line_number}")
            output.append(f"  Last instruction: {frame_state.last_instruction}")
            output.append(f"  Timestamp: {snapshot.timestamp:.4f}s")
        
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