# Hand-Gesture-Controled-Robot


Gesture Controlled Robot Using Computer Vision and ESP32

A real-time robotic system that allows users to control a mobile robot using hand gestures detected by a laptop camera. The system uses deep learning, computer vision, and WiFi communication to convert gestures into robot movement commands.

Project Overview

This project demonstrates a gesture-based human–robot interaction system.

A camera captures the user's hand gesture, which is processed using a pose detection deep learning model. The recognized gesture is then transmitted via WiFi to an ESP32 microcontroller, which controls the robot motors accordingly.

The robot can perform the following actions based on gestures:

👍 Forward

☝️ Left

✌️ Right

✋ Stop


Features

Real-time hand gesture recognition

Wireless robot control using WiFi

Deep learning-based gesture detection

Computer vision using OpenCV

Lightweight and fast pose detection model

ESP32-based robot control



System Architecture

Gesture → Camera → Deep Learning Model → Command → WiFi → ESP32 → Motor Driver → Robot Movement



Hardware Components

ESP32 Microcontroller

L298N Motor Driver

DC Motors

Robot Chassis

Battery Power Supply

Laptop Camera



Software Tools

  Python
  
  OpenCV
  
  YOLOv8 Pose Model
  
  CVAT (Dataset Annotation)
  
  Arduino IDE

Dataset
Images were captured manually and annotated using CVAT with 21 hand keypoints.


Model Training

Model used: YOLOv8n Pose

Training parameters:

Epochs: 150

Image size: 440

Validation images: 89


ESP32 Setup

Open Arduino IDE

Upload the ESP32 motor control code

Connect ESP32 to WiFi

Connect motor driver to ESP32 GPIO pins

Power the robot using battery



Project Structure
gesture-controlled-robot/
│
├── dataset/
│   ├── images
│   ├── labels
│
├── model/
│   ├── best.pt
│
├── esp32/
│   ├── esp32_robot_control.ino
│
├── gesture_control.py
├── label_map.json
├── README.md
