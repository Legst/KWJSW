import sys
from PySide6.QtWidgets import QMainWindow, QApplication

from main_window_ui import Ui_MainWindow


#def convert2QImage(img):
#    height, width, channel = img.shape
#    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
       

    
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    app.exec()