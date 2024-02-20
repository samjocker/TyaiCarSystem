import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
import folium
from folium import plugins
from PyQt5.QtCore import Qt

class MapViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Folium Map in PyQt")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)

        self.setup_map()

    def setup_map(self):
        # Create a Folium map
        # m = folium.Map(location=gps_coordinates, zoom_start=17)
        m = folium.Map(location=[24.99329, 121.32073], zoom_start=17)

        # Add a marker to the map
        # folium.Marker(location=[25.042951, 121.535154], popup="Your Location").add_to(m)

        # Save the map as HTML
        m.save("map.html")

        # Load the HTML into the QtWebEngineView
        self.browser.setHtml(open("map.html").read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapViewer()
    window.show()
    sys.exit(app.exec_())
