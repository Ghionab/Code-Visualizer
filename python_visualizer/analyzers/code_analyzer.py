import ast
import tokenize
import io
import types
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from ..core.data_models import CompilationStage
from ..core.config import Config


class Token:
    """Represents a single token from the tokenizer."""
    
    def __init__(self, type_name: str, string: str, start: tuple, end: tuple, line: str):
        self.type_name = type_name
        self.string = string
        self.start = start  # (line, column)
        self.end = end      # (line, column)
        self.line = line
    
    def __repr__(self):
        return f"Token({self.type_name}, {self.string!r}, {self.start}-{self.end})"


class AnalysisResult:
    """Contains the complete results of code analysis."""
    
    def __init__(self):
        self.source_code: str = ""
        self.tokens: List[Token] = []
        self.ast_tree: Optional[ast.AST] = None
        self.bytecode: Optional[types.CodeType] = None
        self.compilation_stages: List[CompilationStage] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def add_stage(self, stage_name: str, input_data: Any, output_data: Any, explanation: str):
        """Add a compilation stage to the results."""
        stage = CompilationStage(
            stage_name=stage_name,
            input_data=input_data,
            output_data=output_data,
            explanation=explanation
        )
        self.compilation_stages.append(stage)


class CodeAnalyzer:
    """Main entry point for code analysis pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze(self, source: Union[str, Path]) -> AnalysisResult:

        result = AnalysisResult()
        
        try:
            # Step 1: Load source code
            if isinstance(source, (str, Path)) and Path(source).exists():
                # It's a file path
                with open(source, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                result.add_stage("load", str(source), source_code, 
                                f"Loaded source code from file: {source}")
            else:
                # It's source code string
                source_code = str(source)
                result.add_stage("load", "string input", source_code,
                                "Loaded source code from string input")
            
            result.source_code = source_code
            
            # Step 2: Tokenization
            if self.config.analysis.analyze_ast:  # Use AST config as proxy for tokenization
                tokens = self._tokenize(source_code)
                result.tokens = tokens
                result.add_stage("tokenize", source_code, tokens,
                                f"Tokenized source code into {len(tokens)} tokens")
            
            # Step 3: AST parsing
            if self.config.analysis.analyze_ast:
                ast_tree = self._parse_ast(source_code)
                result.ast_tree = ast_tree
                result.add_stage("parse", tokens if tokens else source_code, ast_tree,
                                "Parsed tokens into Abstract Syntax Tree")
            
            # Step 4: Bytecode compilation
            if self.config.analysis.analyze_bytecode:
                bytecode = self._compile_bytecode(ast_tree if ast_tree else source_code)
                result.bytecode = bytecode
                result.add_stage("compile", ast_tree if ast_tree else source_code, bytecode,
                                "Compiled AST into bytecode")
            
        except SyntaxError as e:
            error_msg = f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"
            result.errors.append(error_msg)
            result.add_stage("error", source_code, None, error_msg)
            
        except Exception as e:
            error_msg = f"Analysis Error: {str(e)}"
            result.errors.append(error_msg)
            result.add_stage("error", source_code, None, error_msg)
        
        return result
    
    def _tokenize(self, source: str) -> List[Token]:

        tokens = []
        
        try:
            # Create a StringIO object for the tokenizer
            source_io = io.StringIO(source)
            
            # Tokenize the source
            token_generator = tokenize.generate_tokens(source_io.readline)
            
            for tok in token_generator:
                # Convert tokenize constants to readable names
                type_name = tokenize.tok_name.get(tok.type, f"UNKNOWN({tok.type})")
                
                # Skip ENCODING and ENDMARKER tokens for cleaner output
                if type_name in ('ENCODING', 'ENDMARKER'):
                    continue
                
                token = Token(
                    type_name=type_name,
                    string=tok.string,
                    start=tok.start,
                    end=tok.end,
                    line=tok.line
                )
                tokens.append(token)
                
        except tokenize.TokenError as e:
            # Handle incomplete tokens gracefully
            pass
        
        return tokens
    
    def _parse_ast(self, source: str) -> ast.AST:
        """
        Parse source code into AST using ast module.
        
        Args:
            source: Python source code string
            
        Returns:
            AST tree object
        """
        return ast.parse(source)
    
    def _compile_bytecode(self, tree_or_source: Union[ast.AST, str]) -> types.CodeType:
        """
        Compile AST or source to bytecode.
        
        Args:
            tree_or_source: Either AST tree or source code string
            
        Returns:
            Compiled code object
        """
        if isinstance(tree_or_source, ast.AST):
            return compile(tree_or_source, '<string>', 'exec')
        else:
            return compile(tree_or_source, '<string>', 'exec')
    
    def visualize_tokens(self, tokens: List[Token]) -> str:
        """Create a visual representation of the token stream."""
        if not tokens:
            return "No tokens to display"
        
        output = []
        output.append("=== TOKEN STREAM ===\n")
        
        current_line = 1
        line_tokens = []
        
        for token in tokens:
            # Group tokens by line
            if token.start[0] != current_line:
                if line_tokens:
                    output.append(self._format_line_tokens(current_line, line_tokens))
                current_line = token.start[0]
                line_tokens = []
            
            if token.type_name not in ['NL', 'NEWLINE', 'COMMENT']:
                line_tokens.append(token)
        
        # Handle last line
        if line_tokens:
            output.append(self._format_line_tokens(current_line, line_tokens))
        
        return '\n'.join(output)
    
    def _format_line_tokens(self, line_num: int, tokens: List[Token]) -> str:
        """Format tokens for a single line."""
        if not tokens:
            return ""
        
        output = [f"Line {line_num:2d}: "]
        
        for i, token in enumerate(tokens):
            if i > 0:
                output.append(" → ")
            
            # Format token based on type
            if token.type_name == 'NAME':
                if token.string in ['def', 'class', 'if', 'for', 'while', 'return', 'import']:
                    output.append(f"[KEYWORD:{token.string}]")
                else:
                    output.append(f"[NAME:{token.string}]")
            elif token.type_name == 'NUMBER':
                output.append(f"[NUM:{token.string}]")
            elif token.type_name == 'STRING':
                display_str = token.string[:15] + "..." if len(token.string) > 15 else token.string
                output.append(f"[STR:{display_str}]")
            elif token.type_name == 'OP':
                output.append(f"[OP:{token.string}]")
            else:
                output.append(f"[{token.type_name}:{token.string}]")
        
        return ''.join(output)