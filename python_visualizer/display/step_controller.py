import sys
import time
from typing import Optional, Callable, Set, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

from ..core.data_models import TraceEvent, FrameState
from ..core.config import Config
from .display_engine import DisplayEngine


class ExecutionPhase(Enum):
    """Execution phases for phase navigation."""
    COMPILATION = "compilation"
    INITIALIZATION = "initialization"
    EXECUTION = "execution"
    FUNCTION_CALL = "function_call"
    LOOP_ITERATION = "loop_iteration"
    EXCEPTION_HANDLING = "exception_handling"
    CLEANUP = "cleanup"


class StepMode(Enum):
    """Step modes for controlling execution granularity."""
    INSTRUCTION = "instruction"  # Step through each bytecode instruction
    LINE = "line"                # Step through each source line
    FUNCTION = "function"        # Step through function calls
    PHASE = "phase"              # Step through execution phases


@dataclass
class BreakpointInfo:
    """Information about a breakpoint."""
    line_number: Optional[int] = None
    operation: Optional[str] = None  # Opcode name
    condition: Optional[Callable[[TraceEvent], bool]] = None
    hit_count: int = 0
    enabled: bool = True


class StepController:
    """
    Controller for interactive step-by-step execution.
    
    Provides functionality for:
    - Pause/resume execution
    - Single-step advancement with precise control
    - Breakpoint management for lines and operations
    - Phase navigation to skip to specific execution stages
    """
    
    def __init__(self, display_engine: DisplayEngine, config: Optional[Config] = None):
        """
        Initialize the step controller.
        
        Args:
            display_engine: DisplayEngine instance for output formatting
            config: Optional configuration object
        """
        self.display_engine = display_engine
        self.config = config
        
        # Execution state
        self.is_paused: bool = True  # Start paused for interactive mode
        self.is_running: bool = False
        self.should_stop: bool = False
        self.current_event: Optional[TraceEvent] = None
        self.current_phase: ExecutionPhase = ExecutionPhase.INITIALIZATION
        
        # Step control
        self.step_mode: StepMode = StepMode.LINE
        self.steps_remaining: int = 0  # For multi-step commands
        self.auto_step_delay: float = 0.0  # Delay between auto-steps
        
        # Breakpoints
        self.breakpoints: Dict[int, BreakpointInfo] = {}  # Line breakpoints
        self.operation_breakpoints: Dict[str, BreakpointInfo] = {}  # Opcode breakpoints
        self.conditional_breakpoints: List[BreakpointInfo] = []
        
        # Phase navigation
        self.target_phase: Optional[ExecutionPhase] = None
        self.phase_history: List[ExecutionPhase] = []
        
        # Event history
        self.event_history: List[TraceEvent] = []
        self.current_event_index: int = -1
        
        # Callbacks
        self.on_step_callback: Optional[Callable[[TraceEvent], None]] = None
        self.on_breakpoint_callback: Optional[Callable[[TraceEvent, BreakpointInfo], None]] = None
        self.on_phase_change_callback: Optional[Callable[[ExecutionPhase], None]] = None
    
    def start(self):
        """Start the step controller."""
        self.is_running = True
        self.should_stop = False
        self.is_paused = True  # Start paused for interactive control
        self.current_phase = ExecutionPhase.INITIALIZATION
        self.event_history.clear()
        self.current_event_index = -1
        self.phase_history.clear()
    
    def stop(self):
        """Stop the step controller."""
        self.is_running = False
        self.should_stop = True
        self.is_paused = False
    
    def pause(self):
        """Pause execution."""
        self.is_paused = True
    
    def resume(self):
        """Resume execution."""
        self.is_paused = False
        self.steps_remaining = 0
    
    def step(self, count: int = 1):
        """
        Advance execution by a specified number of steps.
        
        Args:
            count: Number of steps to advance (default: 1)
        """
        self.steps_remaining = count
        self.is_paused = False
    
    def step_over(self):
        """Step over the current line (don't enter function calls)."""
        if self.current_event and self.current_event.frame:
            current_depth = self.current_event.stack_depth
            # Continue until we return to the same or lower depth
            self.step_mode = StepMode.LINE
            self.steps_remaining = 1
            self.is_paused = False
    
    def step_into(self):
        """Step into function calls."""
        self.step_mode = StepMode.INSTRUCTION
        self.steps_remaining = 1
        self.is_paused = False
    
    def step_out(self):
        """Step out of the current function."""
        if self.current_event and self.current_event.frame:
            current_depth = self.current_event.stack_depth
            # Continue until we're at a lower depth (returned from function)
            self.steps_remaining = -1  # Special value for step-out
            self.is_paused = False
    
    def continue_execution(self):
        """Continue execution until next breakpoint or completion."""
        self.is_paused = False
        self.steps_remaining = 0
    
    def set_breakpoint(self, line_number: int, condition: Optional[Callable[[TraceEvent], bool]] = None):
        """
        Set a breakpoint at a specific line number.
        
        Args:
            line_number: Line number to break at
            condition: Optional condition function that must return True to break
        """
        self.breakpoints[line_number] = BreakpointInfo(
            line_number=line_number,
            condition=condition,
            enabled=True
        )
    
    def remove_breakpoint(self, line_number: int):
        """
        Remove a breakpoint at a specific line number.
        
        Args:
            line_number: Line number to remove breakpoint from
        """
        if line_number in self.breakpoints:
            del self.breakpoints[line_number]
    
    def enable_breakpoint(self, line_number: int):
        """Enable a breakpoint."""
        if line_number in self.breakpoints:
            self.breakpoints[line_number].enabled = True
    
    def disable_breakpoint(self, line_number: int):
        """Disable a breakpoint without removing it."""
        if line_number in self.breakpoints:
            self.breakpoints[line_number].enabled = False
    
    def clear_breakpoints(self):
        """Clear all breakpoints."""
        self.breakpoints.clear()
        self.operation_breakpoints.clear()
        self.conditional_breakpoints.clear()
    
    def set_operation_breakpoint(self, operation: str, condition: Optional[Callable[[TraceEvent], bool]] = None):
        """
        Set a breakpoint for a specific bytecode operation.
        
        Args:
            operation: Opcode name (e.g., 'LOAD_CONST', 'CALL_FUNCTION')
            condition: Optional condition function
        """
        self.operation_breakpoints[operation] = BreakpointInfo(
            operation=operation,
            condition=condition,
            enabled=True
        )
    
    def remove_operation_breakpoint(self, operation: str):
        """Remove an operation breakpoint."""
        if operation in self.operation_breakpoints:
            del self.operation_breakpoints[operation]
    
    def add_conditional_breakpoint(self, condition: Callable[[TraceEvent], bool]):
        """
        Add a conditional breakpoint that breaks when condition is True.
        
        Args:
            condition: Function that takes TraceEvent and returns bool
        """
        self.conditional_breakpoints.append(BreakpointInfo(
            condition=condition,
            enabled=True
        ))
    
    def skip_to_phase(self, phase: ExecutionPhase):
        """
        Skip execution to a specific phase.
        
        Args:
            phase: Target execution phase
        """
        self.target_phase = phase
        self.is_paused = False
    
    def set_step_mode(self, mode: StepMode):
        """
        Set the stepping granularity mode.
        
        Args:
            mode: StepMode enum value
        """
        self.step_mode = mode
    
    def set_auto_step(self, delay: float):
        """
        Enable automatic stepping with a delay.
        
        Args:
            delay: Delay in seconds between steps
        """
        self.auto_step_delay = delay
        self.is_paused = False
    
    def should_pause_at_event(self, event: TraceEvent) -> bool:
        """
        Determine if execution should pause at this event.
        
        Args:
            event: TraceEvent to check
            
        Returns:
            True if execution should pause, False otherwise
        """
        # Always pause if explicitly paused
        if self.is_paused and self.steps_remaining <= 0:
            return True
        
        # Check if we should stop
        if self.should_stop:
            return True
        
        # Check line breakpoints
        if event.frame and event.frame.f_lineno in self.breakpoints:
            bp = self.breakpoints[event.frame.f_lineno]
            if bp.enabled:
                # Check condition if present
                if bp.condition is None or bp.condition(event):
                    bp.hit_count += 1
                    if self.on_breakpoint_callback:
                        self.on_breakpoint_callback(event, bp)
                    return True
        
        # Check operation breakpoints
        if event.instruction and event.instruction.opname in self.operation_breakpoints:
            bp = self.operation_breakpoints[event.instruction.opname]
            if bp.enabled:
                if bp.condition is None or bp.condition(event):
                    bp.hit_count += 1
                    if self.on_breakpoint_callback:
                        self.on_breakpoint_callback(event, bp)
                    return True
        
        # Check conditional breakpoints
        for bp in self.conditional_breakpoints:
            if bp.enabled and bp.condition and bp.condition(event):
                bp.hit_count += 1
                if self.on_breakpoint_callback:
                    self.on_breakpoint_callback(event, bp)
                return True
        
        # Check phase navigation
        if self.target_phase is not None:
            current_phase = self._determine_phase(event)
            if current_phase == self.target_phase:
                self.target_phase = None
                return True
        
        # Check step mode
        if self.steps_remaining > 0:
            if self._should_count_as_step(event):
                self.steps_remaining -= 1
                if self.steps_remaining == 0:
                    return True
        
        # Step-out mode
        if self.steps_remaining == -1:
            if self.current_event and event.stack_depth < self.current_event.stack_depth:
                self.steps_remaining = 0
                return True
        
        return False
    
    def process_event(self, event: TraceEvent) -> bool:
        """
        Process an execution event and determine if execution should pause.
        
        Args:
            event: TraceEvent to process
            
        Returns:
            True if execution should pause, False to continue
        """
        # Store event in history
        self.event_history.append(event)
        self.current_event_index = len(self.event_history) - 1
        self.current_event = event
        
        # Update current phase
        new_phase = self._determine_phase(event)
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.phase_history.append(new_phase)
            if self.on_phase_change_callback:
                self.on_phase_change_callback(new_phase)
        
        # Check if we should pause
        should_pause = self.should_pause_at_event(event)
        
        if should_pause:
            self.is_paused = True
            
            # Call step callback
            if self.on_step_callback:
                self.on_step_callback(event)
            
            # Auto-step delay
            if self.auto_step_delay > 0:
                time.sleep(self.auto_step_delay)
                self.is_paused = False
        
        return should_pause
    
    def interactive_step(self, event: TraceEvent):
        """
        Handle interactive stepping with user input.
        
        Args:
            event: Current TraceEvent
        """
        # Display current state
        self._display_current_state(event)
        
        # Wait for user input
        while self.is_paused and not self.should_stop:
            try:
                command = input("\n[Step] (n)ext, (s)tep into, (o)ut, (c)ontinue, (q)uit: ").strip().lower()
                
                if command in ['n', 'next', '']:
                    self.step()
                    break
                elif command in ['s', 'step', 'into']:
                    self.step_into()
                    break
                elif command in ['o', 'out']:
                    self.step_out()
                    break
                elif command in ['c', 'continue']:
                    self.continue_execution()
                    break
                elif command in ['q', 'quit']:
                    self.stop()
                    break
                elif command.startswith('b '):
                    # Set breakpoint
                    try:
                        line_num = int(command.split()[1])
                        self.set_breakpoint(line_num)
                        print(f"Breakpoint set at line {line_num}")
                    except (ValueError, IndexError):
                        print("Invalid breakpoint command. Use: b <line_number>")
                elif command == 'l':
                    # List breakpoints
                    self._list_breakpoints()
                elif command == 'h' or command == 'help':
                    self._show_help()
                else:
                    print(f"Unknown command: {command}. Type 'h' for help.")
            
            except (KeyboardInterrupt, EOFError):
                print("\nExecution interrupted.")
                self.stop()
                break
    
    def get_breakpoints(self) -> List[BreakpointInfo]:
        """Get list of all breakpoints."""
        breakpoints = list(self.breakpoints.values())
        breakpoints.extend(self.operation_breakpoints.values())
        breakpoints.extend(self.conditional_breakpoints)
        return breakpoints
    
    def get_event_history(self) -> List[TraceEvent]:
        """Get the complete event history."""
        return self.event_history.copy()
    
    def get_phase_history(self) -> List[ExecutionPhase]:
        """Get the phase transition history."""
        return self.phase_history.copy()
    
    # Private helper methods
    
    def _determine_phase(self, event: TraceEvent) -> ExecutionPhase:
        """Determine the execution phase from an event."""
        if event.event_type == 'call':
            return ExecutionPhase.FUNCTION_CALL
        elif event.event_type == 'exception':
            return ExecutionPhase.EXCEPTION_HANDLING
        elif event.instruction:
            opname = event.instruction.opname
            if 'FOR_ITER' in opname or 'GET_ITER' in opname:
                return ExecutionPhase.LOOP_ITERATION
        
        return ExecutionPhase.EXECUTION
    
    def _should_count_as_step(self, event: TraceEvent) -> bool:
        """Determine if an event counts as a step based on step mode."""
        if self.step_mode == StepMode.INSTRUCTION:
            return event.instruction is not None
        elif self.step_mode == StepMode.LINE:
            return event.event_type == 'line'
        elif self.step_mode == StepMode.FUNCTION:
            return event.event_type in ['call', 'return']
        elif self.step_mode == StepMode.PHASE:
            # Count phase changes
            return self._determine_phase(event) != self.current_phase
        
        return True
    
    def _display_current_state(self, event: TraceEvent):
        """Display the current execution state."""
        print("\n" + "=" * 80)
        print(self.display_engine.format_execution_step(event))
        
        if event.frame:
            print(f"\nCurrent line: {event.frame.f_lineno}")
            print(f"Function: {event.frame.f_code.co_name}")
            print(f"Phase: {self.current_phase.value}")
            
            # Show local variables
            if event.frame.f_locals:
                print("\nLocal variables:")
                for name, value in list(event.frame.f_locals.items())[:5]:
                    if not name.startswith('__'):
                        print(f"  {name} = {repr(value)[:50]}")
    
    def _list_breakpoints(self):
        """List all breakpoints."""
        print("\nBreakpoints:")
        
        if not self.breakpoints and not self.operation_breakpoints:
            print("  No breakpoints set")
            return
        
        for line_num, bp in sorted(self.breakpoints.items()):
            status = "enabled" if bp.enabled else "disabled"
            hits = f"(hit {bp.hit_count} times)" if bp.hit_count > 0 else ""
            print(f"  Line {line_num}: {status} {hits}")
        
        for op_name, bp in sorted(self.operation_breakpoints.items()):
            status = "enabled" if bp.enabled else "disabled"
            hits = f"(hit {bp.hit_count} times)" if bp.hit_count > 0 else ""
            print(f"  Operation {op_name}: {status} {hits}")
    
    def _show_help(self):
        """Show help message for interactive commands."""
        print("\nInteractive Step Controller Commands:")
        print("  n, next      - Step to next line")
        print("  s, step      - Step into function calls")
        print("  o, out       - Step out of current function")
        print("  c, continue  - Continue execution until breakpoint")
        print("  q, quit      - Stop execution")
        print("  b <line>     - Set breakpoint at line number")
        print("  l            - List all breakpoints")
        print("  h, help      - Show this help message")


def create_step_controller(display_engine: DisplayEngine, config: Optional[Config] = None) -> StepController:
    """
    Create a StepController instance.
    
    Args:
        display_engine: DisplayEngine instance
        config: Optional configuration
        
    Returns:
        StepController instance
    """
    return StepController(display_engine, config)