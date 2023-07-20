from ultralytics import YOLO
import cv2
import supervision as sv
import sys
import asyncio
import os
from azure.iot.device import IoTHubSession
from azure.iot.device import Message
import json
import datetime
import PySpin


START = sv.Point(320,0 )
END = sv.Point(320, 480)
CONNECTION_STRING = "HostName=iot-scus-mvld-nr3.azure-devices.net;DeviceId=myJetson;SharedAccessKey=wb0HD0gCPQnEcuawjzR8PLKJO845uHLj1vRuKI/GOc0="
inCount  = 0
outCount = 0


def main():
    global inCount, outCount
    model = YOLO("beans.pt")

    line_zone = sv.LineZone(start = START, end = END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness = 2, text_thickness = 1, text_scale = 0.5)

    box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness = 1, text_scale = 0.5)

    for result in model.track(source=0, stream=True, show = True, save = True, agnostic_nms=True, spectrum = 'visible'):   
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[detections.class_id == 0]
        labels = [
            f" #{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]
        #print(detections)
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        line_zone.trigger(detections)
        line_zone_annotator.annotate(frame, line_zone)
        inCount = line_zone.in_count 
        outCount = line_zone.out_count
        cv2.imshow("frame", frame)


async def sendMessage():
    print("telemetry")
    async with IoTHubSession.from_connection_string(CONNECTION_STRING) as session:
        print("Connected to IoT Hub")
        msg_txt = '{{"inCount": "{In}", "outCount": "{out}", "time": "{time}", "deviceId": "{deviceId}"}}'
        msg = msg_txt.format(In = inCount, out = outCount, time = datetime.datetime.now(), deviceId = "ManualTestOnNRDell")
        MSG = Message(msg)
        await session.send_message(MSG)
        print(msg)
        await asyncio.sleep(2)
        
if __name__ == "__main__":
    try: 
        main()
    except KeyboardInterrupt:
        print(inCount)
        print(outCount)
        asyncio.run(sendMessage())
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        camera = cam_list[0]
        camera.EndAcquisition()
        camera.DeInit()
        del camera
        cam_list.Clear()
        system.ReleaseInstance() 
        sys.exit(0)


