import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, QUrl

class Bridge(QObject):
    @pyqtSlot(float, float)
    def sendCoordinates(self, lat, lon):
        print(f"[Python] Got coordinates: {lat}, {lon}")
        window.lat_input.setText(str(lat))
        window.lon_input.setText(str(lon))

class MapWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Test")
        self.setGeometry(300, 300, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("Latitude")
        layout.addWidget(self.lat_input)

        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("Longitude")
        layout.addWidget(self.lon_input)

        self.web = QWebEngineView()
        layout.addWidget(self.web)

        self.channel = QWebChannel()
        self.bridge = Bridge()
        self.channel.registerObject("pyjs", self.bridge)
        self.web.page().setWebChannel(self.channel)

        html_path = QUrl.fromLocalFile(str(Path(__file__).with_name("leaflet_map.html")))
        self.web.load(html_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec_())
