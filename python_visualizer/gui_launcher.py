
import sys
import tkinter as tk
from tkinter import messagebox

def check_gui_requirements():
    """Check if GUI requirements are available."""
    try:
        # Test tkinter availability
        root = tk.Tk()
        root.withdraw()  # Hide the test window
        root.destroy()
        return True
    except Exception as e:
        print(f"GUI not available: {e}")
        return False

def launch_gui():
    """Launch the GUI application."""
    if not check_gui_requirements():
        print("Error: GUI components not available.")
        print("Please ensure tkinter is installed with your Python distribution.")
        return 1
    
    try:
        from .gui.main_window import MainWindow
        
        app = MainWindow()
        app.run()
        return 0
        
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(launch_gui())