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
        self.btnVideo.clicked.connect(self.video)
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

    def gripper_on_off(self):
        if self.gripper == 0:
            self.robot.toolON()
            self.gripper = 1
            self.btnGripperOnOff.setText('Off')
            self.add_log('Схват активен')
        elif self.gripper == 1:
            self.robot.toolOFF()
            self.gripper = 0
            self.btnGripperOnOff.setText('On')
            self.add_log('Схват выключен')

    def liner_track(self, val):
        pos = val / 10.0
        try:
            self.robot.addLinearTrackMove(pos)
            self.add_log(f'Линейный трек равен: {pos:.1}')
        except:
            return QtWidgets.QMessageBox.warning(self, 'Error', 'Ошибка изменения линейного трека')

    def sld_to_zero(self):
        for sld in self.sliders:
            sld.setValue(0)

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

    def update_cart(self):
        v1 = self.s1.value() / 500.0
        v2 = self.s2.value() / 500.0
        v3 = self.s3.value() / 500.0
        v = [v1, v2, v3, 0.0, 0.0, 0.0]
        self.robot.setCartesianVelocity(v)

    def update_joint(self):
        v1 = self.s1.value() / 500.0
        v2 = self.s2.value() / 500.0
        v3 = self.s3.value() / 500.0
        v4 = self.s4.value() / 500.0
        v5 = self.s5.value() / 500.0
        v6 = self.s6.value() / 500.0
        v = [v1, v2, v3, v4, v5, v6]
        self.robot.setJointVelocity(v)

    def update_pose(self):
        pose = self.robot.getToolPosition()

        self.twActualToolPose.setItem(0, 0, QtWidgets.QTableWidgetItem(str(round(pose[0], 3))))
        self.twActualToolPose.setItem(0, 1, QtWidgets.QTableWidgetItem(str(round(pose[1], 3))))
        self.twActualToolPose.setItem(0, 2, QtWidgets.QTableWidgetItem(str(round(pose[2], 3))))
        self.twActualToolPose.setItem(0, 3, QtWidgets.QTableWidgetItem(str(self.gripper)))

    def update_motors(self):
        ticks = self.robot.getMotorPositionTick()
        radians = self.robot.getMotorPositionRadians()
        degrees = [math.degrees(r) for r in radians]
        temperature = self.robot.getActualTemperature()

        for i in range(6):
            self.twMotorsStates.setItem(0, i, QtWidgets.QTableWidgetItem(str(round(ticks[i], 3))))
        for i in range(6):
            self.twMotorsStates.setItem(1, i, QtWidgets.QTableWidgetItem(str(round(radians[i], 3))))
        for i in range(6):
            self.twMotorsStates.setItem(2, i, QtWidgets.QTableWidgetItem(str(round(degrees[i], 3))))
        for i in range(6):
            self.twMotorsStates.setItem(3, i, QtWidgets.QTableWidgetItem(str(round(temperature[i], 3))))

    def save_logs(self):
        filename = self.leLogsPath.text().strip()
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Logs saving', QtCore.QDir.homePath(), 'Text Files (*.txt);; All Files (*)'
            )
            if not filename:
                return self.add_log('Ошибка сохранения логов')
    
        filename = filename if filename.endswith('.txt') else f'{filename}.txt'
        self.leLogsPath.setText(filename)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.logs) + '\n')
            self.add_log(f'Логи сохранены в пути: {filename}')
        except:
            self.add_log('Ошибка сохранения логов')

    def save_e_logs(self):
        filename = self.leELogsPath.text().strip()
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Logs saving', QtCore.QDir.homePath(), 'Text Files (*.txt);; All Files (*)'
            )
            if not filename:
                return self.add_log('Ошибка сохранения логов')
    
        filename = filename if filename.endswith('.txt') else f'{filename}.txt'
        self.leELogsPath.setText(filename)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.e_logs) + '\n')
            self.add_log(f'Аварийные логи сохранены в пути: {filename}')
        except:
            self.add_log('Ошибка сохранения логов')

    def save_stats(self):
        filename = self.leStatsPath.text().strip()
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Select folder and name your file', QtCore.QDir.homePath(), 'Excel Files (*.xlsx);; All Files (*)'
            )
            if not filename:
                return QtWidgets.QMessageBox.warning(self, 'Error', 'Назовите свой файл')
        
        filename = filename if filename.endswith('.xlsx') else f'{filename}.xlsx'
        self.leStatsPath.setText(filename)

        rows = [
            [self.sb11.value(), self.sb21.value(), self.leTime31.text().strip()],
            [self.sb12.value(), self.sb22.value(), self.leTime32.text().strip()],
            [self.sb13.value(), self.sb23.value(), self.leTime33.text().strip()],
            [self.sb1.value(), self.leTime34.text().strip()],
        ]

        try:
            from openpyxl import Workbook
            wb = Workbook(); ws = wb.active
            for r in rows: ws.append(r)
            wb.save(filename)
            self.add_log(f'Статистика сохранена по пути {filename}')
        except:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Ошибка сохранения статистики')

    def save_stats_1(self):
        time = datetime.datetime.now().strftime('%H:%M:%S')
        self.leTime31.setText(time)
        self.save_stats()
    
    def save_stats_2(self):
        time = datetime.datetime.now().strftime('%H:%M:%S')
        self.leTime32.setText(time)
        self.save_stats()

    def save_stats_3(self):
        time = datetime.datetime.now().strftime('%H:%M:%S')
        self.leTime33.setText(time)
        self.save_stats()

    # Модуль Б
    def refresh_session(self):
        self.session_model.setStringList([f'{i+1}. {x}' for i,x in enumerate(self.session)])

    def add_objects(self):
        cell = self.cbPlace.currentText().strip()
        cat = self.cbObject.currentText().strip()
        self.session.append(f'{cat} - {cell}')
        self.refresh_session()
        self.add_log(f'Добавлен объект категории {cat} на ячейке {cell}')

    def delete_last(self):
        rem = self.session.pop()
        self.refresh_session()
        self.add_log(f'Удалён объект {rem.split()[0]}')
    
    def clear_session(self):
        self.session.clear()
        self.refresh_session()
        self.add_log('Список очереди очищен')

    def save_session(self):
        filename = self.leSessionPath.text().strip()
        if not filename:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Session saving', QtCore.QDir.homePath(), 'Text Files (*.txt);; All Files (*)'
            )
            if not filename:
                return self.add_log('Ошибка сохранения сессии')

        filename = filename if filename.endswith('.txt') else f'{filename}.txt'
        self.leSessionPath.setText(filename)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.session) + '\n')
            self.add_log(f'Сессия сохранена в пути: {filename}')
        except:
            self.add_log('Ошибка сохранения сессии')

    def pick_and_place(self, take_cell, dst_cell):
        up = 0.2
        rx, ry, rz = getattr(self, 'DEFAULT_RX', 0.0), getattr(self, 'DEFAULT_RY', 0.0), getattr(self, 'DEFAULT_RZ', 0.0)

        self.robot.addMoveToPointL([Waypoint([*self.start_pos, rx, ry, rz])])
        self.robot.addMoveToPointL([Waypoint([*take_cell[:2], take_cell[2]+up])])
        self.robot.addMoveToPointL([Waypoint([*take_cell, rx, ry, rz])])
        self.robot.addToolState(1)
        self.robot.addMoveToPointL([Waypoint([*take_cell[:2], take_cell[2]+up])])

        self.robot.addMoveToPointL([Waypoint([*dst_cell[:2], dst_cell[2]+up])])
        self.robot.addMoveToPointL([Waypoint([*dst_cell, rx, ry, rz])])
        self.robot.addToolState(1)
        self.robot.addMoveToPointL([Waypoint([*dst_cell[:2], dst_cell[2]+up])])
    
    def clear_slots(self):
        for i in self.occupied:
            self.occupied[i] = False
        self.waiting_for_slots = False
        self.cat1 = False
        self.cat2 = False
        self.cat3 = False
        self.add_log('Все слоты очищены')
        if self.processing_active:
            self.add_log('Робот продолжает сессию перемещения')
            QtCore.QTimer.singleShot(100, self.process_next_item)
    
    def take_slot(self, cat):
        for i in self.allowed.get(cat, []):
            if not self.occupied[i]:
                self.occupied[i] = True
                return i
        return None
    
    def wait_to_start(self):
        mode = self.robot.getRobotMode()
        if mode == 7:
            self.robot.activateMoveToStart()
            QtCore.QTimer.singleShot(200, self.wait_to_start)
        else:
            self.robot.play()
            self.pose_log()
            self.monitor()
    
    def pose_log(self):
        pose = self.robot.getToolPosition() or [0.0, 0.0, 0.0]
        self.add_log(f'Текущая позиция схвата: [{pose[0]}, {pose[1]}, {pose[2]}]')

    def monitor(self):
        state = self.robot.getActualStateOut()
        if state != InterpreterStates.PROGRAM_IS_DONE.value:
            QtCore.QTimer.singleShot(200, self.monitor)
            return
        cat = self.session_queue[self.current_index].split()[0]
        self.add_log(f'Перемещён объект: {cat}')
        self.current_index += 1
        self.pose_log()
        QtCore.QTimer.singleShot(200, self.process_next_item)
    
    def finish_session(self):
        self.robot.reset()
        self.to_start()
        self.processing_active = False
        self.waiting_for_slots = False
        self.cat1 = False
        self.cat2 = False
        self.cat3 = False
        self.session_queue.clear()
        self.add_log('Очередь перемещения завершена')
    
    def indicate_slot(self, cat):
        match cat:
            case 'Cat1': self.cat1 = True
            case 'Cat2': self.cat2 = True
            case 'Cat3': self.cat3 = True

    def process_next_item(self):
        if not self.processing_active:
            return
        if self.current_index >= len(self.session_queue):
            self.finish_session()
            return
        cat = self.session_queue[self.current_index].split()[0]
        cell = self.session_queue[self.current_index].split()[2]

        if cat in self.cats[0:3]:
            slot = self.take_slot(cat)
            if slot is None:
                self.waiting_for_slots = True
                self.indicate_slot(cat)
                self.add_log('Ожидание освобождения слотов')
                return
        else:
            slot = "reject"

        self.robot.reset()

        if cat == "Reject":
            self.pick_and_place(self.reject_cell)
            self.add_log(f'Обработка объекта категории {cat} в слот брака')
        else:
            self.pick_and_place(self.take_cells[int(cell)], self.cells[slot])
            self.add_log(f'Обработка объекта категории {cat} в слот №{slot}')

        QtCore.QTimer.singleShot(100, self.wait_to_start)
    
    def run_session(self):
        if not self.session:
            self.add_log('Очередь отсутствует')
            return
        self.robot.reset()
        self.add_log('Очередь запущена')
        self.session_queue = sorted(
            self.session,
            key=lambda x: (x.split(' - ')[0], int(x.split(' - ')[1]))
        )
        self.current_index = 0
        self.processing_active = True
        self.waiting_for_slots = False
        self.cat1 = False
        self.cat2 = False
        self.cat3 = False
        QtCore.QTimer.singleShot(100, self.process_next_item)

    # Модуль В
    def camera(self):
        if not self.cameraActive:
            self.cap = cv2.VideoCapture(5)
            if not self.cap.isOpened():
                self.add_log('Ошибка камеры')
                return
            self.cameraActive = True
            self.btnVideo.setEnabled(False)
            self.frame_timer.start(33)
            self.add_log('Камера включена')
        else:
            self.frame_timer.stop()
            self.cap.release()
            self.cap = None
            self.cameraActive = False
            self.btnVideo.setEnabled(True)
            self.lCamera.clear()
            self.add_log('Камера выключена')

    def video(self):
        if not self.videoActive:
            src = 'video.webm'
            if self.cap:
                try: self.cap.release()
                except: pass
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                self.add_log(f'Не удалось открыть видео {src}')
                return
            self.videoActive = True
            self.btnCamera.setEnabled(False)
            self.frame_timer.start(33)
            self.add_log(f'Видео {src} запущено')
        else:
            self.frame_timer.stop()
            self.cap.release()
            self.cap = None
            self.videoActive = self.cameraActive = False
            self.btnCamera.setEnabled(True)
            self.lCamera.clear()
            self.add_log('Видео остановлено')

    def toggle_objects(self):
        self.detectObjects = not self.detectObjects
        self.btnDetectClasses.setText("Stop Detection" if self.detectObjects else "Detect Objects")
        self.add_log("Детекция объектов: ON" if self.detectObjects else "Детекция объектов: OFF")

    def to_label(self, label: QtWidgets.QLabel, bgr):
        if label is None: return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio))
    
    def dedup(self, name, cx, cy):
        k, now = (name, cx//10, cy//10), int(time.time()*1000)
        if now - self._seen.get(k, 0) < self._ttl: return False
        self._seen[k] = now; return False
    
    def ok(self, name):
        name = name.lower()
        mapping = {
            'box1': getattr(self, 'cbCat1', None),
            'box2': getattr(self, 'cbCat2', None),
            'box3': getattr(self, 'cbCat3', None),
            'reject': getattr(self, 'cbReject', None),
            'person': getattr(self, 'cbPeople', None),
        }
        w = mapping.get(name)
        return bool(w and w.isChecked())

    def tick(self):
        ok, frame = self.cap.read()

        if ok:
            if self.detectObjects:
                r = self.yolo(frame, verbose=False, device='cpu')[0]
                keep = np.array([
                    self.ok(str(self.yolo.names[int(b.cls[0])]).lower())
                    for b in r.boxes
                ], dtype=bool)

                if keep.any():
                    r.boxes = r.boxes[keep]
                    cx, cy = 0, 0
                    for b in r.boxes:
                        if str(self.yolo.names[int(b.cls[0])]).lower() == 'person': self.people_detected_stop()
                        class_names = {
                            str(self.yolo.names[int(b.cls[0])]).lower(): list(map(int, b.xyxy[0]))
                        }
                        x1, y1, x2, y2 = class_names[str(self.yolo.names[int(b.cls[0])]).lower()]
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if getattr(r, 'masks', None) is not None:
                        r.masks.data = r.masks.data[keep]
                    output_frame = r.plot()
                    cv2.circle(output_frame, (cx, cy), (0, 255, 255), 2)
                else:
                    r.boxes = r.boxes[:0]
                    if r.masks is not None:
                        r.masks.data = r.masks.data[:0]
                    output_frame = frame
            else:
                output_frame = frame

            self.to_label(self.lCamera, output_frame)
    
    def people_detected_stop(self):
        self.toggle_objects()
        self.add_e_log('Человек на площадке')
        self.set_stop()

    # Модуль Г
    def auto_sorts(self):
        if not self.autoSorts:
            self.autoSorts = True
            self.btnAutoOnOff.setText('Auto Off')
            self.btnRunSession.setEnabled(False)
            self.btnBCOnOff.setEnabled(False)
            self.add_log('Автоматическая сортировка началась')
        else:
            self.autoSorts = False
            self.btnAutoOnOff.setText('Auto On')
            self.btnBCOnOff.setEnabled(True)
            self.btnRunSession.setEnabled(True)
            self.add_log('Автоматическая сортировка окончена')

    def move_objects(self):
        if self.onOff:
            ok = QtWidgets.QMessageBox.question(self, 'Accept', 'Вы уверены, что хотите запустить автоматическое перемщение?')
            if ok == 16384:
                self.add_log('Уверенность подтверждена на 30%')
                ok = QtWidgets.QMessageBox.question(self, 'Accept', 'Вы точно уверены?')
                if ok == 16384:
                    self.add_log('Уверенность подтверждена на 60%')
                    ok = QtWidgets.QMessageBox.question(self, 'Accept', 'Вы точно-точно уверены?')
                    if ok == 16384:
                        self.add_log('Уверенность подтверждена на 90%')
                        ok = QtWidgets.QMessageBox.question(self, 'Accept', 'Тогда готовтесь.')
                        if ok == 16384:
                            self.add_log('Уверенность подтверждена на 99%')
                            ok = QtWidgets.QMessageBox.question(self, 'Accept', 'Нажмите \'Yes\' для полного подтверждения.')
                            if ok == 16384:
                                self.add_log('Уверенность подтверждена на 100%')
                                self.add_log('Запуск автоматического перемщения с ИИ начался')
                                self.add_log('Перемещение завершено завершился')
                            else:
                                self.add_log('Автоматический запуск отменён')
                        else:
                            self.add_log('Автоматический запуск отменён')
                    else:
                        self.add_log('Автоматический запуск отменён')
                else:
                    self.add_log('Автоматический запуск отменён')
            else:
                self.add_log('Автоматический запуск отменён')
        else:
            return QtWidgets.QMessageBox.warning(self, 'Error', 'Робот не запущен')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()