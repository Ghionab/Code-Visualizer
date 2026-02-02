

import ast
from typing import List, Dict, Any, Optional, Set
from ..core.config import Config


class ControlFlowGraph:
    """Simple control flow graph representation."""
    
    def __init__(self):
        self.nodes = []
        self.conditionals = []
        self.loops = []
        self.functions = []
        self.classes = []
        self.exception_handlers = []
    
    def add_conditional(self, line: int, description: str):
        """Add a conditional construct."""
        self.conditionals.append({"line": line, "description": description})
    
    def add_loop(self, line: int, description: str):
        """Add a loop construct."""
        self.loops.append({"line": line, "description": description})
    
    def add_function(self, line: int, name: str):
        """Add a function definition."""
        self.functions.append({"line": line, "name": name})
    
    def add_class(self, line: int, name: str):
        """Add a class definition."""
        self.classes.append({"line": line, "name": name})
    
    def add_exception_handler(self, line: int, description: str):
        """Add an exception handler."""
        self.exception_handlers.append({"line": line, "description": description})


class ASTAnalysis:
    """Contains the results of AST analysis."""
    
    def __init__(self):
        self.tree: Optional[ast.AST] = None
        self.visualization: str = ""
        self.control_flow: Optional[ControlFlowGraph] = None
        self.node_count: int = 0
        self.depth: int = 0
        self.node_types: Dict[str, int] = {}


class ASTParser:
    """AST parser with visualization and analysis capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def parse(self, source: str) -> ASTAnalysis:
        """
        Parse source code into AST with structural analysis.
        
        Args:
            source: Python source code string
            
        Returns:
            ASTAnalysis containing tree and analysis results
        """
        analysis = ASTAnalysis()
        
        try:
            # Parse the source code
            tree = ast.parse(source)
            analysis.tree = tree
            
            # Analyze the tree structure
            analysis.node_count = self._count_nodes(tree)
            analysis.depth = self._calculate_depth(tree)
            analysis.node_types = self._count_node_types(tree)
            
            # Create visualization
            analysis.visualization = self.visualize_tree(tree)
            
            # Extract control flow if requested
            if self.config.analysis.analyze_control_flow:
                analysis.control_flow = self.extract_control_flow(tree)
            
        except SyntaxError as e:
            analysis.visualization = f"Syntax Error: {e.msg} at line {e.lineno}"
        except Exception as e:
            analysis.visualization = f"AST Analysis Error: {str(e)}"
        
        return analysis
    
    def visualize_tree(self, tree: ast.AST) -> str:
        """
        Create ASCII representation of AST structure.
        
        Args:
            tree: AST tree to visualize
            
        Returns:
            String representation of the tree structure
        """
        if not tree:
            return "No AST tree to display"
        
        output = []
        output.append("=== ABSTRACT SYNTAX TREE ===\n")
        
        # Add tree statistics
        node_count = self._count_nodes(tree)
        depth = self._calculate_depth(tree)
        node_types = self._count_node_types(tree)
        
        output.append(f"Tree Statistics:")
        output.append(f"  Total nodes: {node_count}")
        output.append(f"  Maximum depth: {depth}")
        output.append(f"  Node types: {len(node_types)}")
        output.append("")
        
        # Add node type breakdown
        output.append("Node Type Distribution:")
        for node_type, count in sorted(node_types.items()):
            output.append(f"  {node_type}: {count}")
        output.append("")
        
        # Generate tree visualization
        output.append("Tree Structure:")
        self._visualize_node(tree, output, "", True)
        
        return '\n'.join(output)
    
    def _visualize_node(self, node: ast.AST, output: List[str], prefix: str, is_last: bool):
        """Recursively visualize AST nodes."""
        
        # Determine the connector
        connector = "└── " if is_last else "├── "
        
        # Get node information
        node_info = self._get_node_info(node)
        output.append(f"{prefix}{connector}{node_info}")
        
        # Update prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Get child nodes
        children = list(ast.iter_child_nodes(node))
        
        # Visualize children
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            self._visualize_node(child, output, child_prefix, is_child_last)
    
    def _get_node_info(self, node: ast.AST) -> str:
        """Get descriptive information about an AST node."""
        
        node_type = type(node).__name__
        
        # Add specific information based on node type
        if isinstance(node, ast.Name):
            return f"{node_type}(id='{node.id}')"
        elif isinstance(node, ast.Constant):
            value_repr = repr(node.value)
            if len(value_repr) > 20:
                value_repr = value_repr[:17] + "..."
            return f"{node_type}(value={value_repr})"
        elif isinstance(node, ast.FunctionDef):
            return f"{node_type}(name='{node.name}')"
        elif isinstance(node, ast.ClassDef):
            return f"{node_type}(name='{node.name}')"
        elif isinstance(node, ast.Attribute):
            return f"{node_type}(attr='{node.attr}')"
        elif isinstance(node, ast.BinOp):
            op_name = type(node.op).__name__
            return f"{node_type}(op={op_name})"
        elif isinstance(node, ast.Compare):
            ops = [type(op).__name__ for op in node.ops]
            return f"{node_type}(ops={ops})"
        elif isinstance(node, ast.Call):
            return f"{node_type}()"
        elif isinstance(node, ast.If):
            return f"{node_type}(test)"
        elif isinstance(node, ast.For):
            return f"{node_type}(loop)"
        elif isinstance(node, ast.While):
            return f"{node_type}(loop)"
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            return f"{node_type}(names={names})"
        elif isinstance(node, ast.ImportFrom):
            return f"{node_type}(module='{node.module}')"
        else:
            return node_type
    
    def _count_nodes(self, tree: ast.AST) -> int:
        """Count total number of nodes in the AST."""
        count = 1  # Count the current node
        for child in ast.iter_child_nodes(tree):
            count += self._count_nodes(child)
        return count
    
    def _calculate_depth(self, tree: ast.AST, current_depth: int = 0) -> int:
        """Calculate the maximum depth of the AST."""
        max_depth = current_depth
        for child in ast.iter_child_nodes(tree):
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _count_node_types(self, tree: ast.AST) -> Dict[str, int]:
        """Count occurrences of each node type."""
        counts = {}
        
        def count_recursive(node):
            node_type = type(node).__name__
            counts[node_type] = counts.get(node_type, 0) + 1
            for child in ast.iter_child_nodes(node):
                count_recursive(child)
        
        count_recursive(tree)
        return counts
    
    def extract_control_flow(self, tree: ast.AST) -> ControlFlowGraph:
        """
        Extract control flow information from AST.
        
        Args:
            tree: AST tree to analyze
            
        Returns:
            ControlFlowGraph representing the control flow
        """
        cfg = ControlFlowGraph()
        
        # Find control flow constructs
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                cfg.add_conditional(node.lineno, "if statement")
            elif isinstance(node, ast.For):
                cfg.add_loop(node.lineno, "for loop")
            elif isinstance(node, ast.While):
                cfg.add_loop(node.lineno, "while loop")
            elif isinstance(node, ast.Try):
                cfg.add_exception_handler(node.lineno, "try block")
            elif isinstance(node, ast.FunctionDef):
                cfg.add_function(node.lineno, node.name)
            elif isinstance(node, ast.ClassDef):
                cfg.add_class(node.lineno, node.name)
        
        return cfg