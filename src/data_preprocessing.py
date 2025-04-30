import csv
import gpxpy
import pytz
from datetime import datetime
import sys
import math

import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil
import os


timeStampFile = "20241111_143943/VID_20241111_143943_imu_timestamps.csv"
accelFile = "20241111_143943/VID_20241111_143943accel.csv"
gyroFile = "20241111_143943/VID_20241111_143943gyro.csv"
magneticFile = "20241111_143943/VID_20241111_143943magnetic.csv"

gnssFile = "20241111_143943/20241111-143849.gpx"

# Source data directory
sourceFolder = "20241111_143943/frames"
targetFolder = "20241111_143943/dataset"

# geopose output file
geoposeRes = "20241111_143943/res.csv"

frameRates = 30
startUTCTime = "2024-11-11T06:38:57Z"
startDateTime = datetime.strptime(startUTCTime, "%Y-%m-%dT%H:%M:%SZ")
startDateTime = startDateTime.replace(tzinfo=pytz.utc)
dataFrames = {}


class DataFrame:
    def __init__(self, index, timeStamp, accel, gyro, magnetic, lat, lon, ele) -> None:
        self.index = index
        self.timeStamp = timeStamp
        self.accel = accel
        self.gyro = gyro
        self.magnetic = magnetic
        self.lat = lat
        self.lon = lon
        self.ele = ele


frame_id = 0
index = 0

# Get video timestamp
timeStampList = []
with open(timeStampFile, "r") as csvFile:
    csvreader = csv.reader(csvFile)

    for i, row in enumerate(csvreader):
        if i % frameRates == 0:
            timeStampList.append((i, row[0]))

# Get GNSS data
gnss = []
with open(gnssFile, "r", encoding="utf-8") as f:
    gpx_file = gpxpy.parse(f)

    track = gpx_file.tracks[0]
    segment = track.segments[0]
    for point in segment.points:
        time = point.time
        if time < startDateTime:
            continue
        if time >= startDateTime:
            gnss.append((point.latitude, point.longitude, point.elevation))
            if len(gnss) == len(timeStampList):
                break


def matchBasedonTimestamp(csvFile):
    timeStampIndex = 0
    pre = None
    middle = None
    after = None

    res = []
    for i, row in enumerate(csvreader):
        if i == 0:
            pre = row
            continue
        if i == 1:
            middle = row
            continue

        after = row
        if abs(int(middle[3]) - int(timeStampList[timeStampIndex][1])) <= abs(
            int(after[3]) - int(timeStampList[timeStampIndex][1])
        ) and abs(int(middle[3]) - int(timeStampList[timeStampIndex][1])) <= abs(
            int(pre[3]) - int(timeStampList[timeStampIndex][1])
        ):
            timeStampIndex = timeStampIndex + 1
            res.append(middle[0:3])

        pre = middle
        middle = after
        if len(res) == len(timeStampList):
            break

    if len(res) < len(timeStampList):
        if abs(middle[3] - timeStampList[timeStampIndex][1]) <= abs(
            pre[3] - timeStampList[timeStampIndex][1]
        ):
            res.append(middle[0:3])
    return res


# Read accel data and match it according to timestamp
accel = []

with open(accelFile, "r") as csvFile:
    csvreader = csv.reader(csvFile)

    accel = matchBasedonTimestamp(csvreader)

if len(accel) < len(timeStampList):
    raise IndexError(
        "The length of accel data doesn't match timestamp data! Please reduce the amount of frames!"
    )

# Read gyro data and match it by timestamp
gyro = []

with open(gyroFile, "r") as csvFile:
    csvreader = csv.reader(csvFile)

    gyro = matchBasedonTimestamp(csvreader)

if len(gyro) < len(timeStampList):
    raise IndexError(
        "The length of gyro data doesn't match timestamp data! Please reduce the amount of frames!"
    )

# Read magnetic data and match it according to timestamp
magnetic = []

with open(magneticFile, "r") as csvFile:
    csvreader = csv.reader(csvFile)

    magnetic = matchBasedonTimestamp(csvreader)

if len(magnetic) < len(timeStampList):
    raise IndexError(
        "The length of magnetic data doesn't match timestamp data! Please reduce the amount of frames!"
    )

# Calculate geopose based on IMU


def normalize_quaternion(quaternion):
    """
    Normalized quaternion
    Parameters:
    quaternion (numpy array): quaternion, format is [x, y, z, w]
    Returns:
    norm_quat (numpy array): normalized quaternion
    """
    return quaternion / np.linalg.norm(quaternion)


def norm(qua):
    for i in range(len(timeStampList)):
        q = qua[i]
        n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        q = [q[i] / n for i in range(4)]
        qua[i] = q
    return qua


def compute_initial_orientation(acc, mag):
    """
    Calculate the initial attitude quaternion using accelerometer and magnetometer data
    """
    # Calculating pitch and roll
    gravity_dir = acc / np.linalg.norm(acc)
    pitch = np.arcsin(-gravity_dir[0])
    roll = np.arctan2(gravity_dir[1], gravity_dir[2])

    # Calculate yaw using magnetometer data
    mag_dir = mag / np.linalg.norm(mag)
    yaw = np.arctan2(
        mag_dir[1] * np.cos(roll) - mag_dir[2] * np.sin(roll),
        mag_dir[0] * np.cos(pitch)
        + mag_dir[1] * np.sin(roll) * np.sin(pitch)
        + mag_dir[2] * np.cos(roll) * np.sin(pitch),
    )

    # Convert to quaternion
    initial_orientation = R.from_euler("xyz", [roll, pitch, yaw])
    return initial_orientation


def compute_geopose_quaternion(acc_data, gyro_data, mag_data, dt):
    """
    Combine magnetometer, accelerometer and gyroscope data to calculate GeoPose quaternion at each moment
    Parameters:
    acc_data (numpy array): accelerometer data (n x 3 matrix, each row is [x, y, z] acceleration value)
    gyro_data (numpy array): gyroscope data (n x 3 matrix, each row is [roll, pitch, yaw] angular velocity)
    mag_data (numpy array): magnetometer data (n x 3 matrix, each row is [x, y, z] magnetic value)
    dt (float): time interval (seconds) for each time step

    Returns:
    quaternions (numpy array): calculated attitude quaternion (n x 4 matrix)
    """
    print(np.shape(acc_data))
    n = len(acc_data)
    quaternions = np.zeros((n, 4))

    # Calculate initial attitude using initial acceleration and magnetometer data
    initial_orientation = compute_initial_orientation(acc_data[0], mag_data[0])
    quaternions[0] = normalize_quaternion(initial_orientation.as_quat())

    for i in range(1, n):
        # Calculate rotation increments using gyroscope data
        gyro_rotation = R.from_rotvec(gyro_data[i] * dt)

        # Update Posture
        current_orientation = R.from_quat(quaternions[i - 1]) * gyro_rotation

        # Calibrate yaw angle using magnetometer
        mag_dir = mag_data[i] / np.linalg.norm(mag_data[i])
        yaw = np.arctan2(
            mag_dir[1] * np.cos(current_orientation.as_euler("xyz")[0])
            - mag_dir[2] * np.sin(current_orientation.as_euler("xyz")[0]),
            mag_dir[0] * np.cos(current_orientation.as_euler("xyz")[1])
            + mag_dir[1]
            * np.sin(current_orientation.as_euler("xyz")[0])
            * np.sin(current_orientation.as_euler("xyz")[1])
            + mag_dir[2]
            * np.cos(current_orientation.as_euler("xyz")[0])
            * np.sin(current_orientation.as_euler("xyz")[1]),
        )

        # Update the attitude quaternion and normalize it
        current_orientation = R.from_euler(
            "xyz",
            [
                current_orientation.as_euler("xyz")[0],
                current_orientation.as_euler("xyz")[1],
                yaw,
            ],
        )

        quaternions[i] = normalize_quaternion(current_orientation.as_quat())

    return quaternions


dt = 1

geopose = compute_geopose_quaternion(
    np.array(accel).astype(np.float32),
    np.array(gyro).astype(np.float32),
    np.array(magnetic).astype(np.float32),
    dt,
)
geopose = norm(geopose)

# GNSS log and geopose write to csv file
with open(geoposeRes, "w", newline="") as csvFile:
    writer = csv.writer(csvFile)

    for i in range(len(geopose)):
        writer.writerow((*gnss[i], *geopose[i]))

# Extract images from frames
for i in range(len(timeStampList)):
    sourceFilename = f"frame_{timeStampList[i][0]:04d}.jpg"
    targetFilename = f"frame_{i:04d}.jpg"
    target_path = os.path.join(targetFolder, targetFilename)
    source_path = os.path.join(sourceFolder, sourceFilename)

    shutil.copy2(source_path, target_path)
    print(f"Copied {source_path} to {target_path}")
