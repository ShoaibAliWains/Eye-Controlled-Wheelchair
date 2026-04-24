# Camera-Based Eye-Controlled Wheelchair System

A production-level embedded system using Raspberry Pi 4, OpenCV, and dual BTS7960 motor drivers to control a wheelchair via eye-tracking.

## Hardware Requirements
- Raspberry Pi 4 (Raspberry Pi OS Bullseye Legacy 64-bit recommended)
- Raspberry Pi Camera Module 3 NoIR (or standard webcam)
- 2x BTS7960 43A Motor Drivers
- 24V DC Gear Motors
- IR Ring Light (for stable pupil detection)

## Wiring Guide (Raspberry Pi to BTS7960)
| Motor | RPWM (Forward) | LPWM (Reverse) | R_EN & L_EN | GND |
|-------|----------------|----------------|-------------|-----|
| Left  | GPIO 12 (PWM0) | GPIO 13 (PWM1) | 3.3V or 5V  | Pi GND |
| Right | GPIO 18 (PWM0) | GPIO 19 (PWM1) | 3.3V or 5V  | Pi GND |

*Note: Ensure common ground between Raspberry Pi and Motor Drivers.*

## Installation & Setup

1. **Enable Camera & I2C on Raspberry Pi:**
   ```bash
   sudo raspi-config
   # Go to Interface Options -> Enable Camera
