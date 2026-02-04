import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..core.config import Config, VerbosityLevel, OutputFormat
from ..core.data_models import *


class MainWindow:
    """Main application window for the Python execution visualizer."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.config = Config()
        self.current_file = None
        self.analysis_thread = None
        
        self._setup_window()
        self._create_menu()
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
    def _setup_window(self):
        """Configure the main window properties."""
        self.root.title("Python Code Execution Visualizer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 600)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'bg': '#f0f0f0',
            'code_bg': '#ffffff',
            'output_bg': '#1e1e1e',
            'output_fg': '#ffffff',
            'highlight': '#007acc',
            'error': '#ff4444',
            'success': '#44ff44',
            'warning': '#ffaa44'
        }
        
        self.root.configure(bg=self.colors['bg'])
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open File...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Output...", command=self.save_output, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Code", command=self.clear_code)
        edit_menu.add_command(label="Clear Output", command=self.clear_output)
        edit_menu.add_separator()
        edit_menu.add_command(label="Settings...", command=self.show_settings)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis", command=self.run_analysis, accelerator="F5")
        analysis_menu.add_command(label="Stop Analysis", command=self.stop_analysis, accelerator="Ctrl+C")
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Step Mode", command=self.toggle_step_mode)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show AST", command=lambda: self.toggle_panel('ast'))
        view_menu.add_command(label="Show Bytecode", command=lambda: self.toggle_panel('bytecode'))
        view_menu.add_command(label="Show Memory", command=lambda: self.toggle_panel('memory'))
        view_menu.add_command(label="Show Stack", command=lambda: self.toggle_panel('stack'))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_help)
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        
        # Toolbar
        self.toolbar = ttk.Frame(self.main_frame)
        self._create_toolbar()
        
        # Main content area with paned window
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        
        # Left panel - Code input and controls
        self.left_panel = ttk.Frame(self.paned_window)
        self._create_left_panel()
        
        # Right panel - Analysis output and visualization
        self.right_panel = ttk.Frame(self.paned_window)
        self._create_right_panel()
        
        self.paned_window.add(self.left_panel, weight=1)
        self.paned_window.add(self.right_panel, weight=2)
        
        # Status bar
        self.status_bar = ttk.Frame(self.main_frame)
        self._create_status_bar()
    
    def _create_toolbar(self):
        """Create the toolbar with common actions."""
        
        # File operations
        ttk.Button(self.toolbar, text="Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="Save", command=self.save_output).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Analysis controls
        self.run_button = ttk.Button(self.toolbar, text="▶ Run", command=self.run_analysis)
        self.run_button.pack(side=tk.LEFT, padx=2)
        
        self.stop_button = ttk.Button(self.toolbar, text="⏹ Stop", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        
        self.step_button = ttk.Button(self.toolbar, text="⏭ Step", command=self.step_forward, state=tk.DISABLED)
        self.step_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Verbosity control
        ttk.Label(self.toolbar, text="Verbosity:").pack(side=tk.LEFT, padx=(5, 2))
        self.verbosity_var = tk.StringVar(value="normal")
        verbosity_combo = ttk.Combobox(self.toolbar, textvariable=self.verbosity_var, 
                                     values=["minimal", "normal", "detailed", "expert"],
                                     state="readonly", width=10)
        verbosity_combo.pack(side=tk.LEFT, padx=2)
        verbosity_combo.bind('<<ComboboxSelected>>', self.on_verbosity_changed)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Analysis options
        self.trace_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toolbar, text="Trace", variable=self.trace_var).pack(side=tk.LEFT, padx=2)
        
        self.memory_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toolbar, text="Memory", variable=self.memory_var).pack(side=tk.LEFT, padx=2)
        
        self.ast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toolbar, text="AST", variable=self.ast_var).pack(side=tk.LEFT, padx=2)
    
    def _create_left_panel(self):
        """Create the left panel with code input and controls."""
        
        # Code input section
        code_frame = ttk.LabelFrame(self.left_panel, text="Python Code", padding=5)
        code_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Code editor with line numbers
        editor_frame = ttk.Frame(code_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers
        self.line_numbers = tk.Text(editor_frame, width=4, padx=3, takefocus=0,
                                   border=0, state='disabled', wrap='none',
                                   bg='#f0f0f0', fg='#666666')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Code text area
        self.code_text = scrolledtext.ScrolledText(editor_frame, wrap=tk.NONE, 
                                                  bg=self.colors['code_bg'],
                                                  font=('Consolas', 11))
        self.code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind events for line numbers
        self.code_text.bind('<KeyRelease>', self.update_line_numbers)
        self.code_text.bind('<Button-1>', self.update_line_numbers)
        self.code_text.bind('<MouseWheel>', self.sync_scroll)
        
        # Control panel
        control_frame = ttk.LabelFrame(self.left_panel, text="Execution Control", padding=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Breakpoints
        bp_frame = ttk.Frame(control_frame)
        bp_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(bp_frame, text="Breakpoints:").pack(side=tk.LEFT)
        self.breakpoint_entry = ttk.Entry(bp_frame, width=20)
        self.breakpoint_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(bp_frame, text="Add", command=self.add_breakpoint).pack(side=tk.LEFT, padx=2)
        
        # Breakpoint list
        self.breakpoint_listbox = tk.Listbox(control_frame, height=4)
        self.breakpoint_listbox.pack(fill=tk.X, pady=2)
        
        # Step controls
        step_frame = ttk.Frame(control_frame)
        step_frame.pack(fill=tk.X, pady=2)
        
        self.step_mode_var = tk.BooleanVar()
        ttk.Checkbutton(step_frame, text="Step Mode", variable=self.step_mode_var).pack(side=tk.LEFT)
        
        ttk.Label(step_frame, text="Delay:").pack(side=tk.LEFT, padx=(10, 2))
        self.step_delay_var = tk.DoubleVar(value=0.5)
        step_delay_spin = ttk.Spinbox(step_frame, from_=0.1, to=5.0, increment=0.1,
                                     textvariable=self.step_delay_var, width=8)
        step_delay_spin.pack(side=tk.LEFT, padx=2)
    
    def _create_right_panel(self):
        """Create the right panel with analysis output and visualization."""
        
        # Notebook for different views
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output tab
        self.output_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_frame, text="Output")
        
        self.output_text = scrolledtext.ScrolledText(self.output_frame, 
                                                    bg=self.colors['output_bg'],
                                                    fg=self.colors['output_fg'],
                                                    font=('Consolas', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # AST tab
        self.ast_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ast_frame, text="AST")
        
        self.ast_text = scrolledtext.ScrolledText(self.ast_frame, font=('Consolas', 10))
        self.ast_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bytecode tab
        self.bytecode_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bytecode_frame, text="Bytecode")
        
        self.bytecode_text = scrolledtext.ScrolledText(self.bytecode_frame, font=('Consolas', 10))
        self.bytecode_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Memory tab
        self.memory_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.memory_frame, text="Memory")
        
        # Memory visualization with tree view
        memory_paned = ttk.PanedWindow(self.memory_frame, orient=tk.VERTICAL)
        memory_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Memory tree
        tree_frame = ttk.Frame(memory_paned)
        self.memory_tree = ttk.Treeview(tree_frame, columns=('Type', 'Size', 'RefCount'), show='tree headings')
        self.memory_tree.heading('#0', text='Object')
        self.memory_tree.heading('Type', text='Type')
        self.memory_tree.heading('Size', text='Size')
        self.memory_tree.heading('RefCount', text='Ref Count')
        
        memory_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.memory_tree.yview)
        self.memory_tree.configure(yscrollcommand=memory_scroll.set)
        
        self.memory_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        memory_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Memory details
        details_frame = ttk.Frame(memory_paned)
        self.memory_details = scrolledtext.ScrolledText(details_frame, height=8, font=('Consolas', 9))
        self.memory_details.pack(fill=tk.BOTH, expand=True)
        
        memory_paned.add(tree_frame, weight=2)
        memory_paned.add(details_frame, weight=1)
        
        # Stack tab
        self.stack_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stack_frame, text="Stack")
        
        self.stack_text = scrolledtext.ScrolledText(self.stack_frame, font=('Consolas', 10))
        self.stack_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Execution trace tab
        self.trace_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.trace_frame, text="Execution Trace")
        
        self.trace_text = scrolledtext.ScrolledText(self.trace_frame, font=('Consolas', 9))
        self.trace_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _create_status_bar(self):
        """Create the status bar."""
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_bar, variable=self.progress_var, 
                                          length=200, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Current line indicator
        self.line_label = ttk.Label(self.status_bar, text="Line: 1")
        self.line_label.pack(side=tk.RIGHT, padx=5)
    
    def _setup_layout(self):
        """Setup the main layout."""
        
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.toolbar.pack(fill=tk.X, padx=5, pady=2)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Set initial pane sizes
        self.root.after(100, lambda: self.paned_window.sashpos(0, 400))
    
    def _bind_events(self):
        """Bind keyboard shortcuts and events."""
        
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.save_output())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<F5>', lambda e: self.run_analysis())
        self.root.bind('<Control-c>', lambda e: self.stop_analysis())
        
        # Code editor events
        self.code_text.bind('<KeyRelease>', self.on_code_changed)
        self.memory_tree.bind('<<TreeviewSelect>>', self.on_memory_select)
    
    # Event handlers
    def open_file(self):
        """Open a Python file."""
        file_path = filedialog.askopenfilename(
            title="Open Python File",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.code_text.delete(1.0, tk.END)
                self.code_text.insert(1.0, content)
                self.current_file = file_path
                self.update_line_numbers()
                self.update_status(f"Loaded: {Path(file_path).name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")
    
    def save_output(self):
        """Save the analysis output to a file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.get(1.0, tk.END))
                self.update_status(f"Saved: {Path(file_path).name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def run_analysis(self):
        """Run the code analysis."""
        code = self.code_text.get(1.0, tk.END).strip()
        
        if not code:
            messagebox.showwarning("Warning", "Please enter some Python code to analyze.")
            return
        
        # Update UI state
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.update_status("Running analysis...")
        
        # Clear previous results
        self.clear_output()
        
        # Update configuration from UI
        self.update_config_from_ui()
        
        # Run analysis in separate thread
        self.analysis_thread = threading.Thread(target=self._run_analysis_thread, args=(code,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _run_analysis_thread(self, code):
        """Run analysis in a separate thread."""
        try:
            from ..analyzers.code_analyzer import CodeAnalyzer, AnalysisResult
            from ..analyzers.ast_parser import ASTParser
            from ..analyzers.bytecode_analyzer import BytecodeAnalyzer
            
            # Create analyzers
            code_analyzer = CodeAnalyzer(self.config)
            ast_parser = ASTParser(self.config)
            bytecode_analyzer = BytecodeAnalyzer(self.config)
            
            self.root.after(0, lambda: self.update_progress(10))
            self.root.after(0, lambda: self.append_output("=== Python Code Execution Visualizer ===\n\n"))
            
            # Step 1: Code Analysis
            self.root.after(100, lambda: self.update_progress(25))
            self.root.after(100, lambda: self.append_output("📝 Source Code Analysis:\n"))
            
            analysis_result = code_analyzer.analyze(code)
            
            if analysis_result.errors:
                for error in analysis_result.errors:
                    self.root.after(200, lambda e=error: self.append_output(f"❌ Error: {e}\n"))
                self.root.after(500, lambda: self.analysis_error("Analysis failed due to errors"))
                return
            
            lines = code.splitlines()
            self.root.after(200, lambda: self.append_output(f"   Lines of code: {len(lines)}\n"))
            self.root.after(200, lambda: self.append_output(f"   Characters: {len(code)}\n"))
            self.root.after(200, lambda: self.append_output(f"   Tokens: {len(analysis_result.tokens)}\n\n"))
            
            # Step 2: AST Analysis
            self.root.after(300, lambda: self.update_progress(50))
            self.root.after(300, lambda: self.append_output("🌳 AST Generation:\n"))
            
            if analysis_result.ast_tree:
                ast_analysis = ast_parser.parse(code)
                self.root.after(400, lambda: self.append_output("   Abstract Syntax Tree created successfully\n"))
                self.root.after(400, lambda: self.append_output(f"   Nodes: {ast_analysis.node_count}\n"))
                self.root.after(400, lambda: self.append_output(f"   Depth: {ast_analysis.depth}\n"))
                self.root.after(400, lambda: self.show_real_ast(ast_analysis.visualization))
            
            # Step 3: Bytecode Analysis
            self.root.after(500, lambda: self.update_progress(75))
            self.root.after(500, lambda: self.append_output("⚙️  Bytecode Compilation:\n"))
            
            if analysis_result.bytecode:
                bytecode_analysis = bytecode_analyzer.analyze(analysis_result.bytecode, lines)
                self.root.after(600, lambda: self.append_output("   Bytecode generated successfully\n"))
                self.root.after(600, lambda: self.append_output(f"   Instructions: {len(bytecode_analysis.instructions)}\n"))
                self.root.after(600, lambda: self.show_real_bytecode(bytecode_analysis.visualization))
            
            # Step 4: Token Analysis
            if analysis_result.tokens:
                token_viz = code_analyzer.visualize_tokens(analysis_result.tokens)
                self.root.after(700, lambda: self.show_token_analysis(token_viz))
            
            self.root.after(800, lambda: self.update_progress(90))
            self.root.after(800, lambda: self.append_output("🔍 Analysis Complete:\n"))
            self.root.after(800, lambda: self.append_output("   Ready for detailed exploration\n\n"))
            
            self.root.after(900, lambda: self.update_progress(100))
            self.root.after(900, lambda: self.append_output("✅ All analysis complete! Explore the tabs above.\n"))
            self.root.after(900, lambda: self.analysis_complete())
            
        except Exception as e:
            self.root.after(0, lambda: self.analysis_error(str(e)))
    
    def stop_analysis(self):
        """Stop the running analysis."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            # Note: Python threading doesn't support clean termination
            # In a real implementation, we'd use a flag to signal the thread to stop
            pass
        
        self.analysis_complete()
        self.update_status("Analysis stopped")
    
    def analysis_complete(self):
        """Handle analysis completion."""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.step_button.config(state=tk.NORMAL if self.step_mode_var.get() else tk.DISABLED)
        self.update_status("Analysis complete")
    
    def analysis_error(self, error_msg):
        """Handle analysis error."""
        self.analysis_complete()
        self.append_output(f"❌ Error: {error_msg}\n", color='error')
        messagebox.showerror("Analysis Error", error_msg)
    
    def step_forward(self):
        """Step forward in execution."""
        # TODO: Implement stepping functionality
        self.append_output("⏭ Step forward (not yet implemented)\n")
    
    def clear_code(self):
        """Clear the code editor."""
        self.code_text.delete(1.0, tk.END)
        self.update_line_numbers()
    
    def clear_output(self):
        """Clear all output areas."""
        self.output_text.delete(1.0, tk.END)
        self.ast_text.delete(1.0, tk.END)
        self.bytecode_text.delete(1.0, tk.END)
        self.stack_text.delete(1.0, tk.END)
        self.trace_text.delete(1.0, tk.END)
        self.memory_details.delete(1.0, tk.END)
        
        # Clear memory tree
        for item in self.memory_tree.get_children():
            self.memory_tree.delete(item)
    
    def update_config_from_ui(self):
        """Update configuration from UI controls."""
        self.config.verbosity = VerbosityLevel(self.verbosity_var.get())
        self.config.tracing.trace_opcodes = self.trace_var.get()
        self.config.analysis.analyze_memory_layout = self.memory_var.get()
        self.config.analysis.analyze_ast = self.ast_var.get()
        self.config.interactive.auto_step = not self.step_mode_var.get()
        self.config.interactive.step_delay = self.step_delay_var.get()
    
    def show_real_ast(self, ast_content):
        """Show real AST visualization."""
        self.ast_text.delete(1.0, tk.END)
        self.ast_text.insert(1.0, ast_content)
    
    def show_real_bytecode(self, bytecode_content):
        """Show real bytecode visualization."""
        self.bytecode_text.delete(1.0, tk.END)
        self.bytecode_text.insert(1.0, bytecode_content)
    
    def show_token_analysis(self, token_content):
        """Show token analysis in the output."""
        self.append_output("\n" + "="*50 + "\n")
        self.append_output("TOKEN ANALYSIS:\n")
        self.append_output(token_content)
        self.append_output("\n" + "="*50 + "\n")
    
    def show_ast_placeholder(self, code):
        """Show placeholder AST visualization."""
        ast_content = f"""# Abstract Syntax Tree for:
# {code[:50]}{'...' if len(code) > 50 else ''}

Module(
    body=[
        # AST nodes will be displayed here when implemented
        # This is a placeholder showing the structure
    ],
    type_ignores=[]
)

# Note: Full AST analysis will be available when 
# the AST parser component is implemented (Task 2)."""
        
        self.ast_text.delete(1.0, tk.END)
        self.ast_text.insert(1.0, ast_content)
    
    def show_bytecode_placeholder(self, code):
        """Show placeholder bytecode visualization."""
        bytecode_content = f"""# Bytecode disassembly for:
# {code[:50]}{'...' if len(code) > 50 else ''}

  1           0 LOAD_CONST               0 (<code object>)
              2 LOAD_CONST               1 ('example')
              4 MAKE_FUNCTION            0
              6 STORE_NAME               0 (example)
              8 LOAD_CONST               2 (None)
             10 RETURN_VALUE

# Note: Full bytecode analysis will be available when 
# the bytecode analyzer component is implemented (Task 3).

# Each instruction will include:
# - Opcode name and arguments
# - Stack effects (before/after)
# - Plain-English explanations
# - Source line mappings"""
        
        self.bytecode_text.delete(1.0, tk.END)
        self.bytecode_text.insert(1.0, bytecode_content)
    
    # UI helper methods
    def update_line_numbers(self, event=None):
        """Update line numbers in the code editor."""
        content = self.code_text.get(1.0, tk.END)
        lines = content.count('\n')
        
        line_numbers_content = '\n'.join(str(i) for i in range(1, lines + 1))
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)
        self.line_numbers.insert(1.0, line_numbers_content)
        self.line_numbers.config(state='disabled')
        
        # Update status bar
        cursor_pos = self.code_text.index(tk.INSERT)
        line_num = cursor_pos.split('.')[0]
        self.line_label.config(text=f"Line: {line_num}")
    
    def sync_scroll(self, event):
        """Synchronize scrolling between code and line numbers."""
        self.line_numbers.yview_moveto(self.code_text.yview()[0])
    
    def append_output(self, text, color=None):
        """Append text to the output area."""
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        
        if color and color in self.colors:
            # TODO: Add color formatting
            pass
    
    def update_status(self, message):
        """Update the status bar message."""
        self.status_label.config(text=message)
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_var.set(value)
    
    def add_breakpoint(self):
        """Add a breakpoint."""
        try:
            line_num = int(self.breakpoint_entry.get())
            self.breakpoint_listbox.insert(tk.END, f"Line {line_num}")
            self.breakpoint_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid line number")
    
    def toggle_panel(self, panel_name):
        """Toggle visibility of analysis panels."""
        # TODO: Implement panel toggling
        pass
    
    def toggle_step_mode(self):
        """Toggle step mode."""
        self.step_mode_var.set(not self.step_mode_var.get())
        self.step_button.config(state=tk.NORMAL if self.step_mode_var.get() else tk.DISABLED)
    
    def on_verbosity_changed(self, event):
        """Handle verbosity level change."""
        self.update_status(f"Verbosity set to: {self.verbosity_var.get()}")
    
    def on_code_changed(self, event):
        """Handle code editor changes."""
        self.update_line_numbers()
    
    def on_memory_select(self, event):
        """Handle memory tree selection."""
        selection = self.memory_tree.selection()
        if selection:
            # TODO: Show memory details for selected object
            self.memory_details.delete(1.0, tk.END)
            self.memory_details.insert(1.0, "Memory details will be shown here when memory tracking is implemented.")
    
    def show_settings(self):
        """Show settings dialog."""
        # TODO: Implement settings dialog
        messagebox.showinfo("Settings", "Settings dialog will be implemented in a future update.")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Python Code Execution Visualizer
Version 0.1.0

An educational tool for understanding Python internals through
step-by-step visualization of code execution.

Features:
• Source code analysis and AST visualization
• Bytecode disassembly and explanation
• Memory allocation and reference tracking
• Interactive execution stepping
• Rich graphical interface

Built with Python standard library only."""
        
        messagebox.showinfo("About", about_text)
    
    def show_help(self):
        """Show help dialog."""
        help_text = """Quick Start Guide:

1. Enter Python code in the left panel or open a .py file
2. Click 'Run' to start analysis
3. Use the tabs to explore different views:
   • Output: Main analysis results
   • AST: Abstract Syntax Tree structure
   • Bytecode: Disassembled bytecode with explanations
   • Memory: Object allocation and memory layout
   • Stack: Execution stack visualization
   • Execution Trace: Step-by-step execution log

Interactive Features:
• Set breakpoints by entering line numbers
• Enable step mode for manual execution control
• Adjust verbosity for different detail levels
• Use toolbar checkboxes to enable/disable analysis components

Keyboard Shortcuts:
• Ctrl+O: Open file
• Ctrl+S: Save output
• F5: Run analysis
• Ctrl+C: Stop analysis"""
        
        messagebox.showinfo("Help", help_text)
    
    def run(self):
        """Start the GUI application."""
        # Set some example code
        example_code = '''def fibonacci(n):
    """Generate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        next_fib = fib[i-1] + fib[i-2]
        fib.append(next_fib)
    
    return fib

# Generate first 8 fibonacci numbers
result = fibonacci(8)
print(f"Fibonacci sequence: {result}")'''
        
        self.code_text.insert(1.0, example_code)
        self.update_line_numbers()
        
        self.root.mainloop()


def main():
    """Main entry point for the GUI application."""
    app = MainWindow()
    app.run()


if __name__ == '__main__':
    main()