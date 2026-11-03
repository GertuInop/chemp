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
        self.LVLogs.setModel(self.log_model)

        self.pose = QtCore.QTimer()
        self.pose.timeout.connect(self.update_pose)

        self.motors = QtCore.QTimer()
        self.motors.timeout.connect(self.update_motors)

        self.cartTimer = QtCore.QTimer()
        self.cartTimer.timeout.connect(self.update_cart)
        self.jointTimer = QtCore.QTimer()
        self.jointTimer.timeout.connect(self.update_joint)

        self.current_code = None
        self.lamp_timer = QtCore.QTimer()
        self.lamp_timer.timeout.connect(self.update_code)

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

    # Модуль А
    def add_log(self, msg: str) -> None:
        time = datetime.datetime.now().strftime('[%H:%M:%S]')
        log = f'{time} - {msg}'
        self.logs.append(log)
        self.log_model.setStringList(self.logs)
    
    def add_e_log(self, msg: str) -> None:
        time = datetime.datetime.now().strftime('[%H:%M:%S]')
        log = f'{time} - {msg}'
        self.logs.append(log)
        self.e_logs.append(log)
        self.log_model.setStringList(self.logs)

    def set_code(self, code: str) -> None:
        if self.current_code != code:
            self.current_code = code
        try:
            self.lamp.setLamp(code)
        except:
            return
        
        for i in self.indicators: i.setStyleSheet('')

        match code:
            case '1000': self.indicators[0].setStyleSheet('background-color: blue')
            case '0100': self.indicators[1].setStyleSheet('background-color: green')
            case '0010': self.indicators[2].setStyleSheet('background-color: yellow')
            case '0001': self.indicators[3].setStyleSheet('background-color: red')
        
        if self.people_detected: self.indicators[4].setStyleSheet('background-color: red')
        if self.cat1: self.indicators[5].setStyleSheet('background-color: red')
        if self.cat2: self.indicators[6].setStyleSheet('background-color: red')
        if self.cat3: self.indicators[7].setStyleSheet('background-color: red')

    def compute_code(self):
        try:
            state = self.robot.getActualStateOut()
            if state == InterpreterStates.PROGRAM_STOP_S.value:
                return '0001'
            if state == InterpreterStates.PROGRAM_PAUSE_S.value:
                return '0010'
            if self.waiting_for_slots:
                return '1000'
            if self.processing_active or state == InterpreterStates.IN_TRANSITION.value:
                return '0100'
            return '1000'
        except:
            return '1000'
    
    def update_code(self):
        code = self.compute_code()
        self.set_code(code)

    def on_off(self):
        if self.robot.connect():
            if not self.onOff:
                self.robot.engage()
                self.robot.reset()
                self.lamp_timer.start(300)
                self.pose.start(500)
                self.motors.start(500)
                self.onOff = True
                self.btnOnOff.setText('Off')
                self.add_log('Робот подключен')
            else:
                self.to_start()
                QtCore.QTimer.singleShot(7000, self.shutdown)
    
    def shutdown(self):
        self.lamp_timer.stop()
        self.robot.addConveyerState(0)
        self.robot.conveyer_stop()
        self.pose.stop()
        self.motors.stop()
        self.jointTimer.stop()
        self.cartTimer.stop()
        self.robot.reset()
        self.robot.disengage()
        self.lamp.setLamp('0000')
        for i in self.indicators: i.setStyleSheet('')
        self.onOff = False
        self.pause = False
        self.manualActive = False
        self.btnOnOff.setText('On')
        self.btnPause.setText('Pause')
        self.add_log('Робот выключен')

    def to_start(self):
        state = self.robot.getActualStateOut()
        if not self.in_transition:
            self.robot.moveToInitialPose()
            self.in_transition = True
        elif state == InterpreterStates.IN_TRANSITION:
            QtCore.QTimer.singleShot(200, self.to_start)
            return
        else:
            self.in_transition = False
            self.add_log('Робот на стартовой позиции')

    def set_pause(self):
        if not self.pause:
            self.robot.pause()
            self.pause = True
            self.btnPause.setText('Play')
            self.add_log('Пауза')
        else:
            self.robot.play()
            self.pause = False
            self.btnPause.setText('Pause')
            self.add_log('Робот продолжает выполнять программу')
        
    def set_stop(self):
        self.robot.addConveyerState(0)
        self.robot.conveyer_stop()
        self.lamp_timer.stop()
        self.robot.stop()
        for i in self.indicators: i.setStyleSheet('')
        self.indicators[3].setStyleSheet('background-color: red')
        self.lamp.setLamp('0001')
        self.onOff = False
        self.pause = False
        self.btnOnOff.setText('On')
        self.btnPause.setText('Pause')
        self.add_e_log('Робот аварийно остановлен')

    def start_conveyer(self):
        if not self.conveyer:
            self.robot.conveyer_start()
            self.robot.addConveyerState(1)
            self.conveyer = True
            self.btnStartConveyer.setText('Stop Conveyer')
            self.add_log('Конвейер запущен')
        else:
            self.robot.conveyer_stop()
            self.robot.addConveyerState(0)
            self.conveyer = False
            self.btnStartConveyer.setText('Start Conveyer')
            self.add_log('Конвейер остановлен')

    def m_active(self):
        if not self.manualActive:
            self.btnManualCart.setEnabled(True)
            self.btnManualJoint.setEnabled(True)
            self.manualActive = True
        else:
            self.btnManualCart.setEnabled(False)
            self.btnManualJoint.setEnabled(False)
            self.manualActive = False

    def m_cart(self):
        if not self.cartActive:
            self.robot.manualCartMode()
            self.cartTimer.start(150)
            self.cartActive = True
            self.update_cart()
            self.btnManualJoint.setEnabled(False)
            self.btnManualCart.setText('Stop Cart')
            self.add_log('Ручной режим Декарта включён')
        else:
            self.cartTimer.stop()
            self.cartActive = False
            self.update_cart()
            self.btnManualJoint.setEnabled(True)
            self.btnManualCart.setText('Manual Cart')
            self.add_log('Ручной режим Декарта выключён')
    
    def m_joint(self):
        if not self.jointActive:
            self.robot.manualJointMode()
            self.jointTimer.start(25)
            self.jointActive = True
            self.update_joint()
            self.btnManualCart.setEnabled(False)
            self.btnManualJoint.setText('Stop Joint')
            for sld in self.sliders:
                sld.setEnabled(True)
            self.add_log('Ручной режим Сочленений включён')
        else:
            self.jointTimer.stop()
            self.jointActive = False
            self.update_joint()
            self.btnManualCart.setEnabled(True)
            self.btnManualJoint.setText('Manual Joint')
            for sld in self.sliders:
                sld.setEnabled(False)
            self.add_log('Ручной режим Сочленений выключён')

    def closeEvent(self, event):
        if self.onOff:
            self.to_start()
            self.lamp_timer.stop()
        self.add_log('Приложение завершает работу')
        agreed = QtWidgets.QMessageBox.question(self, 'Accept', 'Вы хотите сохранить логи?')
        if agreed == 16384:
            self.save_logs()
        time.sleep(1)
        self.robot.disengage()
        self.lamp.setLamp('0000')
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()