# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nnGUI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
    QAction, QFileDialog, QApplication)
import torch
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
sns.set(font='SimHei')  # 解决Seaborn中文显示问题

class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 10, 101, 31))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 330, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setEnabled(True)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(270, 100, 271, 71))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setEnabled(True)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setEnabled(True)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.comboBox.setEnabled(True)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.setItemText(0, "")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 2, 1, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox.setEnabled(True)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 1, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setEnabled(True)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(220, 240, 401, 201))
        self.textEdit.setObjectName("textBrowser")
        self.spinBox_2 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_2.setEnabled(True)
        self.spinBox_2.setGeometry(QtCore.QRect(410, 50, 131, 20))
        self.spinBox_2.setObjectName("spinBox_2")
        self.spinBox_3 = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_3.setEnabled(True)
        self.spinBox_3.setGeometry(QtCore.QRect(400, 460, 91, 20))
        self.spinBox_3.setObjectName("spinBox_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setEnabled(True)
        self.label_5.setGeometry(QtCore.QRect(270, 50, 132, 20))
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        self.action_5 = QtWidgets.QAction(MainWindow)
        self.action_5.setObjectName("action_5")
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menu.addSeparator()
        self.menu_2.addAction(self.action_4)
        self.menu_2.addAction(self.action_5)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(320, 200, 91, 21))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(420, 200, 91, 21))
        self.radioButton_2.setObjectName("radioButton_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 500, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setEnabled(True)
        self.label_6.setGeometry(QtCore.QRect(340, 460, 61, 20))
        self.label_6.setObjectName("label_6")

        self.retranslateUi(MainWindow)
        self.label_5.linkActivated['QString'].connect(MainWindow.setWindowTitle)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #这里都是我要绑定的
        self.comboBox.currentTextChanged['QString'].connect(self.get_activation_function)
        self.spinBox.valueChanged['QString'].connect(self.get_num_neural)
        self.spinBox_2.valueChanged['QString'].connect(self.get_random_seed)
        self.action.triggered.connect(self.x_train_in)
        self.action_2.triggered.connect(self.y_train_in)
        self.pushButton.clicked.connect(self.summary_model)
        self.radioButton.clicked.connect(self.get_lossfn)
        self.radioButton_2.clicked.connect(self.get_lossfn)
        self.pushButton_2.clicked.connect(self.start_train)    #开始训练label
        self.spinBox_3.valueChanged['QString'].connect(self.get_epochs)
        self.action_5.triggered.connect(self.get_graph)
        self.action_4.triggered.connect(self.output_model)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "全连接网络工具箱"))
        self.label.setText(_translate("MainWindow", "神经元个数"))
        self.label_3.setText(_translate("MainWindow", "激活函数"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Sigmoid"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Tanh"))
        self.comboBox.setItemText(3, _translate("MainWindow", "ReLU"))
        self.label_4.setText(_translate("MainWindow", "隐含层设置"))
        #self.textEdit.setText('model.summary()')
        self.label_5.setText(_translate("MainWindow", "设置随机树种子"))
        self.pushButton.setText(_translate("MainWindow", "查看模型"))
        self.menu.setTitle(_translate("MainWindow", "导入"))
        self.menu_2.setTitle(_translate("MainWindow", "导出"))
        self.action.setText(_translate("MainWindow", "导入-训练集输入"))
        self.action.setStatusTip('目前仅支持导入 .npy 文件')
        self.action_2.setText(_translate("MainWindow", "导入-训练集输出"))
        self.action_2.setStatusTip('目前仅支持导入 .npy 文件')
        self.action_4.setText(_translate("MainWindow", "导出模型"))
        self.action_5.setText(_translate("MainWindow", "导出学习曲线"))
        self.radioButton.setText(_translate("MainWindow", "分类问题"))
        self.radioButton_2.setText(_translate("MainWindow", "回归问题"))
        self.pushButton_2.setText(_translate("MainWindow", "开始训练"))
        self.label_6.setText(_translate("MainWindow", "训练次数"))



    # 我自己编辑的事件
    def start_train(self):
        print('begin')
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        x = torch.from_numpy(self.x_train).type(torch.float32)
        y = torch.from_numpy(self.y_train).type(torch.float32)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
        train_ds,test_ds = TensorDataset(x_train,y_train),TensorDataset(x_test,y_test)
        train_dl,test_dl = DataLoader(train_ds,shuffle=True),DataLoader(test_ds)
        optim = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.loss,self.val_loss=[],[]
        print('Done')
        self.statusbar.showMessage('正在训练……')
        for epoch in range(self.epochs):
            for x,y in train_ds:
                y_pred = self.model(x)
                loss = self.lossfn(y_pred,y)
                optim.zero_grad()
                loss.backward()
                optim.step()
            print(epoch)
            with torch.no_grad():
                loss=self.lossfn(self.model(x_train),y_train)
                val_loss=self.lossfn(self.model(x_test),y_test)
                self.loss.append(loss)
                self.val_loss.append(val_loss)
                self.textEdit.setText('第 {} 次训练\n训练集损失 {:.4f}\n验证集损失 {:.4f}'.format(epoch,loss,val_loss))
                self.statusbar.showMessage('第 {} 次训练\n训练集损失 {:.4f}\n验证集损失 {:.4f}'.format(epoch,loss,val_loss))

    def get_graph(self):
        sns.set_style('whitegrid')
        plt.plot(self.loss,label='loss')
        plt.plot(self.val_loss,label='val_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()

    def output_model(self):
        print('begin')
        torch.save(self.model, './model.pkl')
        print('Done')

    def get_epochs(self):
        self.epochs=int(self.spinBox_3.text())
        print(self.epochs)

    def get_lossfn(self):
        sender = self.sender()
        if sender.text() == '分类问题':
            self.lossfn = torch.nn.BCEWithLogitsLoss()   #这个是吧BCELoss与sigmoid输出合成一步
            self.type = '分类'
        if sender.text() == '回归问题':
            self.lossfn = torch.nn.MSELoss()
            self.type = '回归'

    def get_activation_function(self):
        self.activation = self.comboBox.currentText()
        print(self.activation)

    def get_num_neural(self):
        self.neurals=self.spinBox.text()
        print(self.neurals)

    def get_random_seed(self):
        self.seed=int(self.spinBox_2.text())
        print(self.seed)

    def x_train_in(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                  "选取文件",
                  "./",
                  "All Files (*);;Text Files (*.txt)")  #设置文件扩展名过滤,注意用双分号间隔
        print(fileName1,filetype)
        ndarray = np.load(fileName1)
        self.x_train=ndarray
        print('x_train have benn loaded!')
        self.feature_shape = ndarray.shape[1]
        print(self.feature_shape)
        print(self.x_train.shape)

    def y_train_in(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                  "选取文件",
                  "./",
                  "All Files (*);;Text Files (*.txt)")  #设置文件扩展名过滤,注意用双分号间隔
        print(fileName1,filetype)
        ndarray = np.load(fileName1)
        print('have loaded')
        self.y_train=ndarray.reshape(-1,1)
        print(self.y_train.shape)

    def summary_model(self):
        #self.feature_shape=100
        print(self.neurals)
        print(self.activation)
        model=self.get_model(self.feature_shape,
                       int(self.neurals),
                       self.activation)
        self.textEdit.setText(str(model))
        self.model = model
        print('Done!')
    def get_model(self,feature_shape,neurals,activation):
        activate={'Sigmoid':torch.nn.Sigmoid(),'Tanh':torch.nn.Tanh(),'ReLU':torch.nn.ReLU()}
        if self.type == '分类':
            model = torch.nn.Sequential(
                torch.nn.Linear(feature_shape,neurals),
                activate[self.activation],
                torch.nn.Linear(neurals,1)
            )
        if self.type == '回归':
            model = torch.nn.Sequential(
                torch.nn.Linear(feature_shape,neurals),
                activate[self.activation],
                torch.nn.Linear(neurals,1)
            )
        return model
#%%
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow  #这两个必须使用

if __name__=='__main__':
    app = QApplication(sys.argv)  # 创建应用程序， 固定的写法
    mainWindow = QMainWindow()
    # 后面都是对构建的主窗口进行操作

    ui = Ui_MainWindow()  #引用类
    ui.setupUi(mainWindow) #类中方法

    mainWindow.show() # 展示
    sys.exit(app.exec_())  # 释放内存

