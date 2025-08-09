import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtChart
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.uic import loadUi
import os
import cv2
import pytesseract
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置为黑体（或其它支持中文的字体）
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


conda_env_path = os.environ.get('CONDA_PREFIX', '')
if conda_env_path:
    plugin_path = str(Path(conda_env_path) / "Lib/site-packages/PyQt5/Qt5/plugins")
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MatrixCalculator(QMainWindow):
    def __init__(self):
        super(MatrixCalculator, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        ui_path = os.path.join(base_dir, "QT.ui")
        loadUi(ui_path, self)
        self.matrix_a = None
        self.matrix_b = None
        self.dist_type = None
        self.dist_params = {}
        self.result = None
        self.setup_connections()

        self.setMinimumSize(self.size())
        
    def setup_connections(self):
        # 设置按钮点击事件
        self.pushButton.clicked.connect(lambda: self.input_matrix('A'))
        self.pushButton_2.clicked.connect(lambda: self.input_matrix('B'))
        self.pushButton_3.clicked.connect(self.generate_random_matrix)
        self.pushButton_4.clicked.connect(self.select_distribution)
        self.pushButton_5.clicked.connect(self.perform_operation)
        self.pushButton_6.clicked.connect(lambda: self.clear_matrix('A'))
        self.pushButton_7.clicked.connect(lambda: self.clear_matrix('B'))
        self.pushButton_8.clicked.connect(self.visualize_matrix)
        self.pushButton_9.clicked.connect(self.clear_result)
        self.pushButton_10.clicked.connect(self.import_from_file) 

    def import_from_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "选择文件",
            "",
            "All Files (*.txt *.csv *.jpg *.png *.jpeg);;Text Files (*.txt);;CSV Files (*.csv);;Image Files (*.jpg *.png *.jpeg)"
        )
        if not file_path:
            return

        try:
            matrix = None
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.txt', '.csv']:
                matrix = np.loadtxt(file_path, delimiter=',' if file_ext == '.csv' else None)
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                
                # Read image and preprocess
                img = cv2.imread(file_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Extract text using OCR
                text = pytesseract.image_to_string(thresh, config='--psm 6')
                
                # Convert text to matrix
                rows = text.strip().split('\n')
                matrix_data = []
                for row in rows:
                    try:
                        # Convert each number in row to float
                        numbers = [float(x) for x in row.strip().split()]
                        if numbers:  # Only add non-empty rows
                            matrix_data.append(numbers)
                    except ValueError:
                        continue
                        
                if matrix_data:
                    matrix = np.array(matrix_data)
                else:
                    raise ValueError("Could not detect valid numbers in image")
            
            if matrix is not None:
                target = self.comboBox_3.currentText()
                if target == 'A':
                    self.matrix_a = matrix
                    self.show_matrix(matrix, self.graphicsView_2)
                else:
                    self.matrix_b = matrix
                    self.show_matrix(matrix, self.graphicsView)
                
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导入文件时出错: {str(e)}")

    def select_distribution(self):
        # 简单实现：弹出对话框让用户选择分布类型和参数
        items = ["均匀分布", "正态分布", "泊松分布"]
        dist_type, ok = QtWidgets.QInputDialog.getItem(self, "选择分布类型", "分布类型：", items, 0, False)
        if not ok:
            return
        self.dist_type = dist_type
        params = {}
        if dist_type == "均匀分布":
            low, ok1 = QtWidgets.QInputDialog.getDouble(self, "参数", "下界 low：", 0)
            high, ok2 = QtWidgets.QInputDialog.getDouble(self, "参数", "上界 high：", 1)
            if ok1 and ok2:
                params['low'] = low
                params['high'] = high
        elif dist_type == "正态分布":
            mean, ok1 = QtWidgets.QInputDialog.getDouble(self, "参数", "均值 mean：", 0)
            std, ok2 = QtWidgets.QInputDialog.getDouble(self, "参数", "标准差 std：", 1)
            if ok1 and ok2:
                params['mean'] = mean
                params['std'] = std
        elif dist_type == "泊松分布":
            lam, ok1 = QtWidgets.QInputDialog.getDouble(self, "参数", "强度参数 lambda：", 1.0, 0.1)
            if ok1:
                params['lambda'] = lam
        self.dist_params = params

    def input_matrix(self, matrix_type):
        dialog = MatrixInputDialog(self)
        if dialog.exec_():
            matrix = dialog.get_matrix()
            if matrix_type == 'A':
                self.matrix_a = matrix
                self.show_matrix(matrix, self.graphicsView_2)
            else:
                self.matrix_b = matrix
                self.show_matrix(matrix, self.graphicsView)

    def generate_random_matrix(self):
        rows = self.spinBox_2.value()
        cols = self.spinBox_3.value()
        target = self.comboBox.currentText()
        if rows <= 0 or cols <= 0:
            QMessageBox.warning(self, "错误", "行数和列数必须大于0")
            return
        if not self.dist_type:
            QMessageBox.warning(self, "错误", "请先选择分布类型（点击分布类型按钮）")
            return
        try:
            if self.dist_type == "均匀分布":
                low = self.dist_params.get('low', 0)
                high = self.dist_params.get('high', 1)
                matrix = np.random.uniform(low, high, (rows, cols))
            elif self.dist_type == "正态分布":
                mean = self.dist_params.get('mean', 0)
                std = self.dist_params.get('std', 1)
                matrix = np.random.normal(mean, std, (rows, cols))
            elif self.dist_type == "泊松分布":
                lam, ok1 = QtWidgets.QInputDialog.getDouble(self, "参数", "强度参数 lambda:", 1.0, 0.1)
                if ok1:
                    matrix = np.random.poisson(lam, (rows, cols))
            else:
                matrix = np.random.rand(rows, cols)
            if target == 'A':
                self.matrix_a = matrix
                self.show_matrix(matrix, self.graphicsView_2)
            else:
                self.matrix_b = matrix
                self.show_matrix(matrix, self.graphicsView)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"生成矩阵时出错: {str(e)}")
    def perform_operation(self):
        operation = self.comboBox_2.currentText()
        try:
            self.result = None
            if operation == "加法":
                self.result = self.matrix_a + self.matrix_b
            elif operation == "减法":
                self.result = self.matrix_a - self.matrix_b
            elif operation == "乘法":
                self.result = np.dot(self.matrix_a, self.matrix_b)
            elif operation == "转置":
                self.result = np.transpose(self.matrix_a if self.matrix_a is not None else self.matrix_b)
            elif operation == "逆矩阵":
                matrix = self.matrix_a if self.matrix_a is not None else self.matrix_b
                self.result = np.linalg.inv(matrix)
            elif operation == "行列式":
                matrix = self.matrix_a if self.matrix_a is not None else self.matrix_b
                self.result = np.linalg.det(matrix)
            elif operation == "特征值":
                matrix = self.matrix_a if self.matrix_a is not None else self.matrix_b
                eigenvalues = np.linalg.eigvals(matrix)
                # Convert eigenvalues to a 2D array for display
                self.result = np.array([[val] for val in eigenvalues])
            elif operation == "特征向量":
                matrix = self.matrix_a if self.matrix_a is not None else self.matrix_b
                eigenvalues, eigenvectors = np.linalg.eig(matrix)
                self.result = eigenvectors
            elif operation == "幂":
                matrix = self.matrix_a if self.matrix_a is not None else self.matrix_b
                power, ok = QtWidgets.QInputDialog.getInt(self, "设置幂次", "请输入幂次:", 2, -100, 100, 1)
                if ok:
                    self.result = np.linalg.matrix_power(matrix, power)
                else:
                    return

            self.show_matrix(self.result, self.graphicsView_3)
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    def show_matrix(self, matrix, view):
        # Create table widget
        table = QTableWidget()
        if isinstance(matrix, (int, float)):
            table.setRowCount(1)
            table.setColumnCount(1)
            table.setItem(0, 0, QTableWidgetItem(f"{matrix:.2f}"))
        else:
            matrix = np.array(matrix)
            if matrix.ndim == 1:
                rows = matrix.shape[0]
                table.setRowCount(rows)
                table.setColumnCount(1)
                for i in range(rows):
                    table.setItem(i, 0, QTableWidgetItem(f"{matrix[i]:.2f}"))
            else:
                rows, cols = matrix.shape
                table.setRowCount(rows)
                table.setColumnCount(cols)
            
            # Fill table with matrix values
            for i in range(rows):
                for j in range(cols):
                    item = QTableWidgetItem(f"{matrix[i,j]:.2f}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)  # Center align text
                    table.setItem(i, j, item)
        
        # Adjust table appearance
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        
        # Center the table in view
        table.setMinimumSize(table.horizontalHeader().length() + 20,
                            table.verticalHeader().length() + 20)
        table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        table.verticalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)

        
        # Create scene and add table
        scene = view.scene()
        if scene is None:
            scene = QtWidgets.QGraphicsScene()
            view.setScene(scene)
        scene.clear()
        scene.addWidget(table)

    def clear_matrix(self, matrix_type):
        if matrix_type == 'A':
            self.matrix_a = None
            self.graphicsView_2.scene().clear()
            self.graphicsView_5.scene().clear()
        else:
            self.matrix_b = None
            self.graphicsView.scene().clear()
            self.graphicsView_4.scene().clear()

    def clear_result(self):
        self.graphicsView_3.scene().clear()
        self.graphicsView_6.scene().clear()

    def visualize_matrix(self):
        if self.matrix_a is None and self.matrix_b is None:
            QMessageBox.warning(self, "错误", "请先输入至少一个矩阵")
            return

        if self.matrix_a is not None:
            fig_a = plt.figure(figsize=(8, 3))
            
            # 2D bar plot
            ax_a1 = fig_a.add_subplot(121)
            im = ax_a1.imshow(self.matrix_a, cmap='viridis')
            # For matrices 10x10 or smaller, show values
            if self.matrix_a.shape[0] <= 10 and self.matrix_a.shape[1] <= 10:
                for i in range(self.matrix_a.shape[0]):
                    for j in range(self.matrix_a.shape[1]):
                        ax_a1.text(j, i, f'{self.matrix_a[i,j]:.1f}', 
                                 ha='center', va='center')
            plt.colorbar(im, ax=ax_a1)
            ax_a1.set_title('矩阵 A (2D)')
            
            # 3D bar plot
            ax_a2 = fig_a.add_subplot(122, projection='3d')
            x_len, y_len = self.matrix_a.shape
            x, y = np.meshgrid(range(x_len), range(y_len))
            ax_a2.bar3d(x.flatten(), y.flatten(), 
                    np.zeros_like(self.matrix_a).flatten(),
                    1, 1, self.matrix_a.flatten())
            ax_a2.set_title('矩阵 A (3D)')
            
            plt.tight_layout()
            canvas_a = FigureCanvas(fig_a)
            scene_a = QtWidgets.QGraphicsScene()
            self.graphicsView_5.setScene(scene_a)
            scene_a.addWidget(canvas_a)
            
        if self.matrix_b is not None:
            fig_b = plt.figure(figsize=(8, 3))
            
            # 2D bar plot
            ax_b1 = fig_b.add_subplot(121)
            im = ax_b1.imshow(self.matrix_b, cmap='viridis')
            if self.matrix_b.shape[0] <= 10 and self.matrix_b.shape[1] <= 10:
                for i in range(self.matrix_b.shape[0]):
                    for j in range(self.matrix_b.shape[1]):
                        ax_b1.text(j, i, f'{self.matrix_b[i,j]:.1f}', 
                                 ha='center', va='center')
            plt.colorbar(im, ax=ax_b1)
            ax_b1.set_title('矩阵 B (2D)')
            
            # 3D bar plot
            ax_b2 = fig_b.add_subplot(122, projection='3d')
            x_len, y_len = self.matrix_b.shape
            x, y = np.meshgrid(range(x_len), range(y_len))
            ax_b2.bar3d(x.flatten(), y.flatten(), 
                    np.zeros_like(self.matrix_b).flatten(),
                    1, 1, self.matrix_b.flatten())
            ax_b2.set_title('矩阵 B (3D)')
            
            plt.tight_layout()
            canvas_b = FigureCanvas(fig_b)
            scene_b = QtWidgets.QGraphicsScene()
            self.graphicsView_4.setScene(scene_b)
            scene_b.addWidget(canvas_b)

        if hasattr(self, 'graphicsView_6') and self.result is not None:
            if isinstance(self.result, (int, float)):
                result_matrix = np.array([[self.result]])
            else:
                result_matrix = self.result

            fig_result = plt.figure(figsize=(12, 3))
            
            # 2D visualization
            ax_result1 = fig_result.add_subplot(131)
            im = ax_result1.imshow(result_matrix, cmap='viridis')
            if result_matrix.shape[0] <= 10 and result_matrix.shape[1] <= 10:
                for i in range(result_matrix.shape[0]):
                    for j in range(result_matrix.shape[1]):
                        ax_result1.text(j, i, f'{result_matrix[i,j]:.1f}', 
                                      ha='center', va='center')
            plt.colorbar(im, ax=ax_result1)
            ax_result1.set_title('运算结果 (2D)')
            
            # 3D bar visualization
            ax_result2 = fig_result.add_subplot(132, projection='3d')
            x_len, y_len = result_matrix.shape
            x, y = np.meshgrid(range(x_len), range(y_len))
            ax_result2.bar3d(x.flatten(), y.flatten(),
                            np.zeros_like(result_matrix).flatten(),
                            1, 1, result_matrix.flatten())
            ax_result2.set_title('运算结果 (3D)')

            # Eigenvectors visualization for 3x3 matrix
            if self.comboBox_2.currentText() == "特征向量" and result_matrix.shape == (3, 3):
                ax_eigen = fig_result.add_subplot(133, projection='3d')
                eigenvals, eigenvecs = np.linalg.eig(result_matrix)
                
                ax_eigen.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5, length=1)
                ax_eigen.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, length=1)
                ax_eigen.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5, length=1)
                
                colors = ['red', 'green', 'blue']
                for i, (eigenval, eigenvec) in enumerate(zip(eigenvals, eigenvecs.T)):
                    ax_eigen.quiver(0, 0, 0, 
                                  eigenvec[0], eigenvec[1], eigenvec[2],
                                  color=colors[i], 
                                  label=f'λ={eigenval:.2f}')
                
                ax_eigen.set_title('特征向量')
                ax_eigen.legend()
                ax_eigen.set_box_aspect([1,1,1])

            plt.tight_layout()
            canvas_result = FigureCanvas(fig_result)
            scene_result = QtWidgets.QGraphicsScene()
            self.graphicsView_6.setScene(scene_result)
            scene_result.addWidget(canvas_result)

class MatrixInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("输入矩阵")
        layout = QVBoxLayout()
        
        # 使用文本框替代表格
        self.text_input = QtWidgets.QTextEdit()
        self.text_input.setPlaceholderText("请输入矩阵，用空格分隔列，换行分隔行\n例如：\n1 2 3\n4 5 6")
        
        # 添加确定和取消按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | 
            QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(self.text_input)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_matrix(self):
        try:
            # 获取文本并按行分割
            text = self.text_input.toPlainText().strip()
            rows = text.split('\n')
            
            # 将每行文本转换为数字列表
            matrix = []
            for row in rows:
                numbers = [float(x) for x in row.strip().split()]
                matrix.append(numbers)
            
            # 检查每行长度是否相同
            if not all(len(row) == len(matrix[0]) for row in matrix):
                raise ValueError("所有行的长度必须相同")
            
            return np.array(matrix)
            
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"输入格式错误: {str(e)}")
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calculator = MatrixCalculator()
    calculator.show()
    app.setStyle('Fusion')  # 设置应用程序的样式
    from PyQt5.QtGui import QFont
from PIL import Image
app.setFont(QFont("SimHei"))
sys.exit(app.exec_())
