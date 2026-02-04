

import dis
import types
from typing import List, Dict, Any, Optional, Tuple
from ..core.data_models import EnhancedInstruction
from ..core.config import Config


class BytecodeAnalysis:
    """Contains the results of bytecode analysis."""
    
    def __init__(self):
        self.instructions: List[EnhancedInstruction] = []
        self.source_mapping: Dict[int, List[EnhancedInstruction]] = {}
        self.visualization: str = ""
        self.statistics: Dict[str, Any] = {}


class BytecodeAnalyzer:
    """Bytecode analyzer with disassembly and educational explanations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.opcode_explanations = self._init_opcode_explanations()
    
    def analyze(self, code: types.CodeType, source_lines: Optional[List[str]] = None) -> BytecodeAnalysis:
        """
        Analyze bytecode and create comprehensive analysis.
        
        Args:
            code: Compiled code object
            source_lines: Optional source code lines for mapping
            
        Returns:
            BytecodeAnalysis containing disassembly and analysis
        """
        analysis = BytecodeAnalysis()
        
        try:
            # Disassemble the bytecode
            instructions = self.disassemble(code)
            analysis.instructions = instructions
            
            # Create source mapping
            if source_lines:
                analysis.source_mapping = self.map_to_source(instructions, source_lines)
            
            # Generate visualization
            analysis.visualization = self.visualize_bytecode(instructions, source_lines)
            
            # Calculate statistics
            analysis.statistics = self._calculate_statistics(instructions)
            
        except Exception as e:
            analysis.visualization = f"Bytecode Analysis Error: {str(e)}"
        
        return analysis
    
    def disassemble(self, code: types.CodeType) -> List[EnhancedInstruction]:
        """
        Disassemble bytecode using dis module.
        
        Args:
            code: Compiled code object
            
        Returns:
            List of enhanced instruction objects
        """
        instructions = []
        
        try:
            # Get bytecode instructions
            for instruction in dis.get_instructions(code):
                enhanced = EnhancedInstruction(
                    instruction=instruction,
                    stack_effect=self._get_stack_effect(instruction),
                    explanation=self.explain_opcode(instruction),
                    source_line=None,  # Will be filled by source mapping
                    related_objects=self._get_related_objects(instruction, code)
                )
                instructions.append(enhanced)
                
        except Exception as e:
            # Handle disassembly errors gracefully
            pass
        
        return instructions
    
    def explain_opcode(self, instruction: dis.Instruction) -> str:
        """
        Provide plain-English explanation of opcode.
        
        Args:
            instruction: Bytecode instruction
            
        Returns:
            Human-readable explanation of what the instruction does
        """
        opname = instruction.opname
        arg = instruction.arg
        argval = instruction.argval
        
        # Get base explanation
        base_explanation = self.opcode_explanations.get(opname, f"Execute {opname} operation")
        
        # Add specific details based on instruction
        if opname == 'LOAD_CONST':
            return f"Load constant value {repr(argval)} onto the stack"
        elif opname == 'LOAD_NAME':
            return f"Load variable '{argval}' onto the stack"
        elif opname == 'STORE_NAME':
            return f"Store top stack value in variable '{argval}'"
        elif opname == 'LOAD_GLOBAL':
            return f"Load global variable '{argval}' onto the stack"
        elif opname == 'STORE_GLOBAL':
            return f"Store top stack value in global variable '{argval}'"
        elif opname == 'LOAD_FAST':
            return f"Load local variable '{argval}' onto the stack"
        elif opname == 'STORE_FAST':
            return f"Store top stack value in local variable '{argval}'"
        elif opname == 'LOAD_ATTR':
            return f"Load attribute '{argval}' from object on stack"
        elif opname == 'STORE_ATTR':
            return f"Store value in attribute '{argval}' of object"
        elif opname == 'CALL_FUNCTION':
            return f"Call function with {arg} positional arguments"
        elif opname == 'CALL_FUNCTION_KW':
            return f"Call function with {arg} arguments (some keyword)"
        elif opname == 'CALL_FUNCTION_EX':
            flags = arg or 0
            if flags & 1:
                return "Call function with *args and **kwargs"
            else:
                return "Call function with *args"
        elif opname == 'JUMP_FORWARD':
            return f"Jump forward {arg} bytes to instruction {instruction.offset + 2 + arg}"
        elif opname == 'JUMP_IF_TRUE_OR_POP':
            return f"Jump to {argval} if top of stack is true, otherwise pop it"
        elif opname == 'JUMP_IF_FALSE_OR_POP':
            return f"Jump to {argval} if top of stack is false, otherwise pop it"
        elif opname == 'POP_JUMP_IF_TRUE':
            return f"Pop stack and jump to {argval} if value is true"
        elif opname == 'POP_JUMP_IF_FALSE':
            return f"Pop stack and jump to {argval} if value is false"
        elif opname == 'COMPARE_OP':
            return f"Compare top two stack values using '{argval}' operator"
        elif opname == 'BINARY_ADD':
            return "Add top two stack values (TOS + TOS1)"
        elif opname == 'BINARY_SUBTRACT':
            return "Subtract top two stack values (TOS1 - TOS)"
        elif opname == 'BINARY_MULTIPLY':
            return "Multiply top two stack values (TOS1 * TOS)"
        elif opname == 'BINARY_TRUE_DIVIDE':
            return "Divide top two stack values (TOS1 / TOS)"
        elif opname == 'BINARY_MODULO':
            return "Modulo top two stack values (TOS1 % TOS)"
        elif opname == 'BINARY_POWER':
            return "Raise TOS1 to power of TOS (TOS1 ** TOS)"
        elif opname == 'BUILD_LIST':
            return f"Build list from top {arg} stack items"
        elif opname == 'BUILD_TUPLE':
            return f"Build tuple from top {arg} stack items"
        elif opname == 'BUILD_SET':
            return f"Build set from top {arg} stack items"
        elif opname == 'BUILD_MAP':
            return f"Build dictionary from top {arg * 2} stack items"
        elif opname == 'FOR_ITER':
            return f"Iterate over TOS, jump to {argval} when exhausted"
        elif opname == 'GET_ITER':
            return "Get iterator from TOS object"
        elif opname == 'RETURN_VALUE':
            return "Return TOS to function caller"
        elif opname == 'POP_TOP':
            return "Remove top stack item"
        elif opname == 'DUP_TOP':
            return "Duplicate top stack item"
        elif opname == 'ROT_TWO':
            return "Swap top two stack items"
        elif opname == 'ROT_THREE':
            return "Rotate top three stack items"
        else:
            return base_explanation
    
    def map_to_source(self, instructions: List[EnhancedInstruction], 
                     source_lines: List[str]) -> Dict[int, List[EnhancedInstruction]]:
        """
        Map bytecode instructions to source lines.
        
        Args:
            instructions: List of enhanced instructions
            source_lines: Source code lines
            
        Returns:
            Dictionary mapping line numbers to instructions
        """
        mapping = {}
        
        for enhanced_instr in instructions:
            instr = enhanced_instr.instruction
            if instr.starts_line is not None:
                line_num = instr.starts_line
                if line_num <= len(source_lines):
                    enhanced_instr.source_line = source_lines[line_num - 1].strip()
                    
                    if line_num not in mapping:
                        mapping[line_num] = []
                    mapping[line_num].append(enhanced_instr)
        
        return mapping
    
    def visualize_bytecode(self, instructions: List[EnhancedInstruction], 
                          source_lines: Optional[List[str]] = None) -> str:
        """
        Create a visual representation of the bytecode.
        
        Args:
            instructions: List of enhanced instructions
            source_lines: Optional source code lines
            
        Returns:
            Formatted string showing bytecode disassembly
        """
        if not instructions:
            return "No bytecode to display"
        
        output = []
        output.append("=== BYTECODE DISASSEMBLY ===\n")
        
        # Add statistics
        stats = self._calculate_statistics(instructions)
        output.append(f"Instructions: {stats['total_instructions']}")
        output.append(f"Unique opcodes: {stats['unique_opcodes']}")
        output.append(f"Stack depth changes: {stats['total_stack_effect']}")
        output.append("")
        
        # Group instructions by source line
        current_line = None
        
        for enhanced_instr in instructions:
            instr = enhanced_instr.instruction
            
            # Show source line if it changed
            if instr.starts_line != current_line and instr.starts_line is not None:
                current_line = instr.starts_line
                if source_lines and current_line <= len(source_lines):
                    source_line = source_lines[current_line - 1].strip()
                    output.append(f"\n{current_line:3d}  {source_line}")
                else:
                    output.append(f"\n{current_line:3d}  <source line not available>")
            
            # Format instruction
            offset_str = f"{instr.offset:4d}"
            opname_str = f"{instr.opname:<20}"
            
            # Format argument
            if instr.arg is not None:
                arg_str = f"{instr.arg:4d}"
                if instr.argval is not None and instr.argval != instr.arg:
                    argval_str = repr(instr.argval)
                    if len(argval_str) > 30:
                        argval_str = argval_str[:27] + "..."
                    arg_str += f" ({argval_str})"
            else:
                arg_str = ""
            
            # Stack effect
            stack_effect = enhanced_instr.stack_effect
            if stack_effect > 0:
                stack_str = f"[+{stack_effect}]"
            elif stack_effect < 0:
                stack_str = f"[{stack_effect}]"
            else:
                stack_str = "[0]"
            
            # Combine instruction line
            instr_line = f"     {offset_str} {opname_str} {arg_str:<20} {stack_str}"
            output.append(instr_line)
            
            # Add explanation if verbose
            if self.config.verbosity.value in ['detailed', 'expert']:
                explanation = enhanced_instr.explanation
                output.append(f"          ↳ {explanation}")
        
        return '\n'.join(output)
    
    def _get_stack_effect(self, instruction: dis.Instruction) -> int:
        """Calculate the stack effect of an instruction."""
        try:
            return dis.stack_effect(instruction.opcode, instruction.arg)
        except (ValueError, SystemError):
            # Some instructions may not have computable stack effects
            return 0
    
    def _get_related_objects(self, instruction: dis.Instruction, code: types.CodeType) -> List[Any]:
        """Get objects related to the instruction (constants, names, etc.)."""
        related = []
        
        if instruction.argval is not None:
            related.append(instruction.argval)
        
        # Add context from code object
        if instruction.opname in ['LOAD_CONST', 'LOAD_GLOBAL', 'LOAD_NAME']:
            if hasattr(code, 'co_names') and instruction.opname in ['LOAD_GLOBAL', 'LOAD_NAME']:
                if instruction.arg < len(code.co_names):
                    related.append(code.co_names[instruction.arg])
            elif hasattr(code, 'co_consts') and instruction.opname == 'LOAD_CONST':
                if instruction.arg < len(code.co_consts):
                    related.append(code.co_consts[instruction.arg])
        
        return related
    
    def _calculate_statistics(self, instructions: List[EnhancedInstruction]) -> Dict[str, Any]:
        """Calculate statistics about the bytecode."""
        if not instructions:
            return {}
        
        opcodes = [instr.instruction.opname for instr in instructions]
        stack_effects = [instr.stack_effect for instr in instructions]
        
        return {
            'total_instructions': len(instructions),
            'unique_opcodes': len(set(opcodes)),
            'most_common_opcode': max(set(opcodes), key=opcodes.count) if opcodes else None,
            'total_stack_effect': sum(stack_effects),
            'max_stack_growth': max(stack_effects) if stack_effects else 0,
            'max_stack_shrink': min(stack_effects) if stack_effects else 0,
            'opcode_distribution': {opcode: opcodes.count(opcode) for opcode in set(opcodes)}
        }
    
    def _init_opcode_explanations(self) -> Dict[str, str]:
        """Initialize base explanations for opcodes."""
        return {
            # Stack manipulation
            'POP_TOP': 'Remove the top item from the stack',
            'ROT_TWO': 'Swap the top two items on the stack',
            'ROT_THREE': 'Rotate the top three items on the stack',
            'DUP_TOP': 'Duplicate the top item on the stack',
            'DUP_TOP_TWO': 'Duplicate the top two items on the stack',
            
            # Unary operations
            'UNARY_POSITIVE': 'Apply unary + to top of stack',
            'UNARY_NEGATIVE': 'Apply unary - to top of stack',
            'UNARY_NOT': 'Apply logical NOT to top of stack',
            'UNARY_INVERT': 'Apply bitwise NOT to top of stack',
            
            # Binary operations
            'BINARY_POWER': 'Raise TOS1 to the power of TOS',
            'BINARY_MULTIPLY': 'Multiply TOS1 by TOS',
            'BINARY_MATRIX_MULTIPLY': 'Matrix multiply TOS1 by TOS',
            'BINARY_FLOOR_DIVIDE': 'Floor divide TOS1 by TOS',
            'BINARY_TRUE_DIVIDE': 'Divide TOS1 by TOS',
            'BINARY_MODULO': 'Calculate TOS1 modulo TOS',
            'BINARY_ADD': 'Add TOS1 and TOS',
            'BINARY_SUBTRACT': 'Subtract TOS from TOS1',
            'BINARY_SUBSCR': 'Get TOS1[TOS]',
            'BINARY_LSHIFT': 'Left shift TOS1 by TOS',
            'BINARY_RSHIFT': 'Right shift TOS1 by TOS',
            'BINARY_AND': 'Bitwise AND of TOS1 and TOS',
            'BINARY_XOR': 'Bitwise XOR of TOS1 and TOS',
            'BINARY_OR': 'Bitwise OR of TOS1 and TOS',
            
            # In-place operations
            'INPLACE_POWER': 'In-place power operation',
            'INPLACE_MULTIPLY': 'In-place multiply operation',
            'INPLACE_MATRIX_MULTIPLY': 'In-place matrix multiply',
            'INPLACE_FLOOR_DIVIDE': 'In-place floor divide',
            'INPLACE_TRUE_DIVIDE': 'In-place true divide',
            'INPLACE_MODULO': 'In-place modulo operation',
            'INPLACE_ADD': 'In-place add operation',
            'INPLACE_SUBTRACT': 'In-place subtract operation',
            'INPLACE_LSHIFT': 'In-place left shift',
            'INPLACE_RSHIFT': 'In-place right shift',
            'INPLACE_AND': 'In-place bitwise AND',
            'INPLACE_XOR': 'In-place bitwise XOR',
            'INPLACE_OR': 'In-place bitwise OR',
            
            # Store/delete subscripts
            'STORE_SUBSCR': 'Store TOS in TOS1[TOS2]',
            'DELETE_SUBSCR': 'Delete TOS1[TOS]',
            
            # Coroutine operations
            'GET_AWAITABLE': 'Get awaitable from TOS',
            'GET_AITER': 'Get async iterator from TOS',
            'GET_ANEXT': 'Get next item from async iterator',
            'BEFORE_ASYNC_WITH': 'Prepare for async with statement',
            
            # Miscellaneous
            'PRINT_EXPR': 'Print expression result',
            'SET_LINENO': 'Set line number for debugging',
            'RAISE_VARARGS': 'Raise exception with arguments',
            'RETURN_VALUE': 'Return value from function',
            'IMPORT_STAR': 'Import all names from module',
            'SETUP_ANNOTATIONS': 'Setup annotations dictionary',
            'YIELD_VALUE': 'Yield value from generator',
            'POP_BLOCK': 'Pop block from block stack',
            'END_ASYNC_FOR': 'End async for loop',
            'POP_EXCEPT': 'Pop exception from exception stack',
            
            # Variable access
            'STORE_NAME': 'Store value in name',
            'DELETE_NAME': 'Delete name',
            'STORE_GLOBAL': 'Store value in global variable',
            'DELETE_GLOBAL': 'Delete global variable',
            'LOAD_CONST': 'Load constant value',
            'LOAD_NAME': 'Load name value',
            'LOAD_GLOBAL': 'Load global variable',
            'LOAD_FAST': 'Load local variable',
            'STORE_FAST': 'Store value in local variable',
            'DELETE_FAST': 'Delete local variable',
            
            # Attribute access
            'LOAD_ATTR': 'Load attribute from object',
            'STORE_ATTR': 'Store value in object attribute',
            'DELETE_ATTR': 'Delete object attribute',
            
            # Comparison operations
            'COMPARE_OP': 'Compare two values',
            
            # Import operations
            'IMPORT_NAME': 'Import module',
            'IMPORT_FROM': 'Import name from module',
            
            # Jump operations
            'JUMP_FORWARD': 'Jump forward unconditionally',
            'JUMP_IF_TRUE_OR_POP': 'Jump if true or pop',
            'JUMP_IF_FALSE_OR_POP': 'Jump if false or pop',
            'POP_JUMP_IF_TRUE': 'Pop and jump if true',
            'POP_JUMP_IF_FALSE': 'Pop and jump if false',
            'JUMP_ABSOLUTE': 'Jump to absolute position',
            
            # For loop operations
            'FOR_ITER': 'Iterate over sequence',
            'GET_ITER': 'Get iterator from object',
            
            # Function calls
            'CALL_FUNCTION': 'Call function',
            'CALL_FUNCTION_KW': 'Call function with keyword args',
            'CALL_FUNCTION_EX': 'Call function with extended args',
            'CALL_METHOD': 'Call method',
            
            # Building operations
            'BUILD_TUPLE': 'Build tuple from stack items',
            'BUILD_LIST': 'Build list from stack items',
            'BUILD_SET': 'Build set from stack items',
            'BUILD_MAP': 'Build dictionary from stack items',
            'BUILD_CONST_KEY_MAP': 'Build dict with constant keys',
            'BUILD_STRING': 'Build string from stack items',
            'BUILD_SLICE': 'Build slice object',
            
            # Exception handling
            'SETUP_EXCEPT': 'Setup exception handler',
            'SETUP_FINALLY': 'Setup finally block',
            'SETUP_WITH': 'Setup with statement',
            'SETUP_ASYNC_WITH': 'Setup async with statement',
            
            # Extended operations
            'EXTENDED_ARG': 'Extended argument for next opcode',
            'LIST_APPEND': 'Append to list',
            'SET_ADD': 'Add to set',
            'MAP_ADD': 'Add key-value pair to map',
        }