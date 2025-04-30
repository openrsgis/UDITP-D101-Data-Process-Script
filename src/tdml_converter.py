import json
import os
import csv

# dataset metadata


def tdmlLabel(type, id, dataURL, labels=None, dataTime=None, geopose=None):
    data = {"type": type, "id": id, "dataURL": dataURL}
    if labels is not None:
        data["labels"] = labels

    if dataTime is not None:
        data["dataTime"] = dataTime

    if geopose is not None:
        data["geopose"] = geopose

    return data


if __name__ == "__main__":
    dataNum = 47
    label_file_json = "20241111_143943/labels.json"

    imageList = [f"dataset/frame_{i:04d}.jpg" for i in range(dataNum)]
    dataList = []

    gnssDataList = []
    gnssData = "20241111_143943/res.csv"
    with open(gnssData, "r") as csvFile:
        csvreader = csv.reader(csvFile)
        for row in csvreader:
            log = {
                "position": {
                    "lon": float(row[1]),
                    "lat": float(row[0]),
                    "h": float(row[2]),
                },
                "quaternion": {
                    "x": float(row[3]),
                    "y": float(row[4]),
                    "z": float(row[5]),
                    "w": float(row[6]),
                },
            }
            gnssDataList.append(log)

    tdmlFormatLabels = {
        "type": "AI_EOTrainingDataset",
        "id": "whu_example2",
        "name": "WHU-example2",
        "description": "Example dataset for UDTIP.",
        "version": "1.0",
        "amountOfTrainingData": dataNum,
        "createdTime": "2024-11-11",
        "providers": ["Wuhan University"],
        "classes": [{"key": "Asphalt Road", "value": None}],
        "numberOfClasses": 1,
        "bands": [
            {"name": [{"code": "red"}]},
            {"name": [{"code": "green"}]},
            {"name": [{"code": "blue"}]},
        ],
        "tasks": [
            {
                "type": "AI_EOTask",
                "id": "whu_example",
                "taskType": "Scene Classification",
            }
        ],
    }

    for i in range(dataNum):
        dataList.append(
            tdmlLabel(
                "AI_EOTrainingData",
                f"road_{i:04d}",
                [f"{imageList[i]}"],
                [{"type": "AI_SceneLabel", "class": "Asphalt Road"}],
                "2024-11-11",
                gnssDataList[i],
            )
        )

    tdmlFormatLabels["data"] = dataList
    with open(label_file_json, "w") as jsonfile:
        json.dump(tdmlFormatLabels, jsonfile, indent=4)
