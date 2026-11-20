"""Launch the desktop assistant from the Windows shortcut or command line."""

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from assistant_app.window import AssistantWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Local Voice Assistant")
    icon_path = Path(__file__).with_name("assets") / "voice-assistant.ico"
    app.setWindowIcon(QIcon(str(icon_path)))
    window = AssistantWindow()
    window.show()
    sys.exit(app.exec())
