"""Rich-text conversation display and plain text for speech."""

from PySide6.QtGui import (
    QDesktopServices, QPalette, QTextBlockFormat, QTextCharFormat,
    QTextCursor, QTextDocument, QTextOption,
)
from PySide6.QtWidgets import QTextBrowser

MARKDOWN = (
    QTextDocument.MarkdownFeature.MarkdownDialectGitHub
    | QTextDocument.MarkdownFeature.MarkdownNoHTML
)


def spoken_text(markdown: str) -> str:
    document = QTextDocument()
    document.setMarkdown(markdown, MARKDOWN)
    return document.toPlainText().replace("*", "").strip()


class TranscriptView(QTextBrowser):
    def __init__(self):
        super().__init__(openLinks=False)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.anchorClicked.connect(self.open_source)

    def open_source(self, url):
        if url.scheme() in ("https", "http"):
            QDesktopServices.openUrl(url)

    def append_message(self, role, text):
        label = "You" if role == "user" else "Assistant"
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.document().isEmpty():
            heading = QTextBlockFormat()
            heading.setTopMargin(12)
            cursor.insertBlock(heading, QTextCharFormat())
        bold = QTextCharFormat()
        bold.setFontWeight(700)
        cursor.insertText(label, bold)
        cursor.insertBlock(QTextBlockFormat(), QTextCharFormat())
        if role == "assistant":
            first_block = cursor.block()
            cursor.insertMarkdown(text, MARKDOWN)
            self.style_reply_links(first_block)
        else:
            cursor.insertText(text, QTextCharFormat())
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def style_reply_links(self, block):
        link_style = QTextCharFormat()
        link_style.setForeground(self.palette().brush(QPalette.ColorRole.Text))
        link_style.setFontUnderline(True)
        document = self.document()
        while block.isValid():
            fragments = block.begin()
            links = []
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if fragment.charFormat().isAnchor():
                    links.append((fragment.position(), fragment.length()))
                fragments += 1
            for position, length in links:
                cursor = QTextCursor(document)
                cursor.setPosition(position)
                cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(link_style)
            block = block.next()
