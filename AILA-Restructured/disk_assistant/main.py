import tkinter as tk
from config import load_config
from ui.gui import DiskAssistantGUI
from utils.logger import logger


def main():
    """Main entry point for the Disk Assistant application"""
    try:
        # Load configuration
        load_config()
        logger.info("Starting Disk Assistant...")

        # Create root window
        root = tk.Tk()
        
        # Initialize the main application
        app = DiskAssistantGUI(root)

        # Set up close handler
        root.protocol("WM_DELETE_WINDOW", app.on_close)

        # Start the application
        logger.info("Application started successfully")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error starting application: {e}")
        raise


if __name__ == "__main__":
    main()