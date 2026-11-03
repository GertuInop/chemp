from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from motion.core import RobotControl, LedLamp, Waypoint
from motion.robot_control import InterpreterStates
import sys, os, math, design, time, datetime, cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    DEFAULT_RX = math.pi / 2.0
    DEFAULT_RY = 0
    DEFAULT_RZ = math.pi

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.robot = RobotControl('192.168.2.100')
        self.lamp = LedLamp('192.168.2.101')

        # Модуль А
        self.gripper = 0
        self.onOff = False
        self.pause = False
        self.conveyer = False
        self.manualActive = False
        self.cartActive = False
        self.jointActive = False

        self.indicators = [self.lWait, self.lWork, self.lPause, self.lStop, self.lPeopleDetected, self.lRow1, self.lRow2, self.lRow3]
        self.sliders = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6]

        self.btnOnOff.clicked.connect(self.on_off)
        self.btnPause.clicked.connect(self.set_pause)
        self.btnToStart.clicked.connect(self.to_start)
        self.btnStop.clicked.connect(self.set_stop)
        self.btnStartConveyer.clicked.connect(self.start_conveyer)
        self.btnToStart.clicked.connect(self.to_start)
        self.btnManualActive.clicked.connect(self.m_active)
        self.btnManualCart.clicked.connect(self.m_cart)
        self.btnManualJoint.clicked.connect(self.m_joint)
        self.btnGripperOnOff.clicked.connect(self.gripper_on_off)
        self.btnSaveStats.clicked.connect(self.save_stats)
        self.btnSaveLogs.clicked.connect(self.save_logs)
        self.btnSaveELogs.clicked.connect(self.save_e_logs)

        self.sldLinerTrack.valueChanged.connect(self.liner_track)
        
        self.s1.sliderReleased.connect(self.sld_to_zero)
        self.s2.sliderReleased.connect(self.sld_to_zero)
        self.s3.sliderReleased.connect(self.sld_to_zero)
        self.s4.sliderReleased.connect(self.sld_to_zero)
        self.s5.sliderReleased.connect(self.sld_to_zero)
        self.s6.sliderReleased.connect(self.sld_to_zero)

        self.sb11.valueChanged.connect(self.save_stats_1)
        self.sb21.valueChanged.connect(self.save_stats_1)
        self.sb12.valueChanged.connect(self.save_stats_2)
        self.sb22.valueChanged.connect(self.save_stats_2)
        self.sb13.valueChanged.connect(self.save_stats_3)
        self.sb23.valueChanged.connect(self.save_stats_3)

        self.logs = []
        self.e_logs = []
        self.log_model = QtCore.QStringListModel()
        self.lvLogs.setModel(self.log_model)

        self.pose = QtCore.QTimer()
        self.pose.timeout.connect(self.update_pose)

        self.motors = QtCore.QTimer()
        self.motors.timeout.connect(self.update_motors)

        self.cartTimer = QtCore.QTimer()
        self.cartTimer.timeout.connect(self.update_cart)
        self.jointTimer = QtCore.QTimer()
        self.jointTimer.timeout.connect(self.update_joint)

        # Модуль Б
        self.processing_active = False
        self.waiting_for_slots = False
        self.people_detected = False
        self.cat1 = False
        self.cat2 = False
        self.cat3 = False
        self.in_transition = False

        self.session = []
        self.session_model = QtCore.QStringListModel()
        self.lvSession.setModel(self.session_model)

        self.start_pos = [0.0, 0.0, 0.0]
        self.take_cells = {
            1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0], 
            4: [0.0, 0.0, 0.0], 5: [0.0, 0.0, 0.0], 6: [0.0, 0.0, 0.0], 
            7: [0.0, 0.0, 0.0], 8: [0.0, 0.0, 0.0], 9: [0.0, 0.0, 0.0]
        }
        self.takeTrack = 0.0
        self.cells = {
            1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0], 
            4: [0.0, 0.0, 0.0], 5: [0.0, 0.0, 0.0], 6: [0.0, 0.0, 0.0], 
            7: [0.0, 0.0, 0.0], 8: [0.0, 0.0, 0.0], 9: [0.0, 0.0, 0.0]
        }
        self.cellTracks = {1:1,2:1}

        self.rejectCell = [1.05, -0.17, -0.11]
        self.rejectTrack = 0.0

        self.cats = ['Cat1', 'Cat2', 'Cat3', 'Reject']
        self.allowed = {'Cat1': [1,2,3], 'Cat2': [4,5,6], 'Cat3': [7,8,9]}
        self.occupied = {i: False for i in range(1,10)}
        self.session_queue = []
        self.current_index = 0

        self.btnAddObject.clicked.connect(self.add_objects)
        self.btnDeleteLast.clicked.connect(self.delete_last)
        self.btnClearSession.clicked.connect(self.clear_session)
        self.btnClearAllSlots.clicked.connect(self.clear_slots)
        self.btnRunSession.clicked.connect(self.run_session)
        self.btnSaveSession.clicked.connect(self.save_session)

        # Модуль В
        self.btnCamera.clicked.connect(self.camera)
        self.btnDetectClasses.clicked.connect(self.toggle_objects)

        self.yolo = YOLO('yolo26n-seg.pt')
        self.cap = None
        self.current_name = None
        self._seen, self._ttl = {}, 200
        self.videoActive = False
        self.cameraActive = False
        self.detectObjects = False
        self.det_objects = {}
        self.first_saw = False

        self.frame_timer = QtCore.QTimer()
        self.frame_timer.timeout.connect(self.tick)

        self.autoSorts = False

        # Модуль Г
        self.btnAutoOnOff.clicked.connect(self.auto_sorts)

        self.add_log('Приложение запущено')