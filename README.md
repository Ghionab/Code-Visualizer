# Python Code Execution Visualizer

An educational tool that provides step-by-step visualization and explanation of Python code execution, from source code parsing through runtime behavior.

## Features

- **Graphical User Interface**: Rich GUI with tabbed views, syntax highlighting, and interactive controls
- **Source Code Analysis**: Tokenization, AST parsing, and bytecode compilation visualization
- **Execution Tracing**: Step-by-step monitoring of bytecode execution with stack state tracking
- **Memory Visualization**: Object allocation tracking, reference counting, and memory layout diagrams
- **Control Flow Analysis**: Detailed explanation of conditionals, loops, and exception handling
- **Interactive Stepping**: Pause, resume, and breakpoint functionality for educational exploration
- **Rich Console Output**: Colorful, well-structured output with ASCII diagrams and plain-English explanations

## Requirements

- Python 3.10 or higher
- No external dependencies (uses only Python standard library)

## Installation

```bash
pip install python-execution-visualizer
```

Or clone and install from source:

```bash
git clone <repository-url>
cd python-execution-visualizer
pip install -e .
```

## Usage

### Graphical Interface

Launch the GUI for an interactive visual experience:

```bash
# Launch GUI directly
python-visualizer --gui

# Or use the dedicated GUI launcher
python-visualizer-gui
```

### Command Line Interface

```bash
# Visualize a Python file
python-visualizer script.py

# Visualize inline code
python-visualizer -c "print('Hello, World!')"

# Interactive stepping mode
python-visualizer --step script.py

# Detailed explanations
python-visualizer --verbose script.py
```

### Advanced Options

```bash
# Disable specific analysis components
python-visualizer --no-memory --no-ast script.py

# Set breakpoints
python-visualizer --breakpoint 10 --breakpoint 25 script.py

# Output to file
python-visualizer -o output.txt script.py

# Plain text output (no colors)
python-visualizer --no-color script.py
```

## Development Status

This project is currently under development. The following components are implemented:

- [x] Project structure and core interfaces
- [ ] Code analyzer and AST parser
- [ ] Bytecode analyzer
- [ ] Execution tracer
- [ ] Memory tracker
- [ ] Control flow analyzer
- [ ] Display engine
- [ ] Step controller
- [ ] Integration and CLI enhancement

## Contributing

This is an educational project. Contributions are welcome!

## License

MIT License - see LICENSE file for details.