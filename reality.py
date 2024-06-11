# 동영상을 입력받아 인식
# 아두이노와 시리얼 통신 o

import cv2
import numpy as np
import serial

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    black_only = cv2.bitwise_and(frame, frame, mask=mask)
    
    gray = cv2.cvtColor(black_only, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, 0),
        (width, 0),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, maxLineGap=50)
    
    if lines is not None:
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
        
        def average_lines(lines):
            if len(lines) == 0:
                return None
            x_coords = []
            y_coords = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            poly = np.polyfit(x_coords, y_coords, 1)
            x_start = int(np.min(x_coords))
            x_end = int(np.max(x_coords))
            y_start = int(poly[0] * x_start + poly[1])
            y_end = int(poly[0] * x_end + poly[1])
            return [[x_start, y_start, x_end, y_end]]
        
        left_line = average_lines(left_lines)
        right_line = average_lines(right_lines)
        
        if left_line is not None and right_line is not None:
            for x1, y1, x2, y2 in left_line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # 빨간색
            for x1, y1, x2, y2 in right_line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # 빨간색
            
            x1_left, y1_left, x2_left, y2_left = left_line[0]
            x1_right, y1_right, x2_right, y2_right = right_line[0]
            
            left_bottom = (x1_left + x2_left) // 2
            right_bottom = (x1_right + x2_right) // 2
            bottom_center = (left_bottom + right_bottom) // 2
            
            left_top = (x1_left + x2_left) // 2
            right_top = (x1_right + x2_right) // 2
            top_center = (left_top + right_top) // 2
            
            cv2.line(frame, (bottom_center, height), (top_center, 0), (0, 255, 0), 3)  # 초록색
            
            return bottom_center, top_center, width // 2
    
    return None, None, None

# 동영상 파일 경로
video_path = 'park.mp4'

# 동영상 파일을 입력으로 받기
cap = cv2.VideoCapture(video_path)

# 아두이노와 직렬 통신 설정 (포트를 라즈베리 파이에 연결된 아두이노의 포트로 설정)
ser = serial.Serial('COM3', 9600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    bottom_center, top_center, frame_center = process_frame(frame)
    
    if bottom_center is not None and top_center is not None:
        if bottom_center < frame_center - 10:
            ser.write(b"left\n")
            print("left")
        elif bottom_center > frame_center + 10:
            ser.write(b"right\n")
            print("right")
        else:
            ser.write(b"forward\n")
            print("forward")
    else:
        ser.write(b"forward\n")
        print("stop")
    
    cv2.imshow('Processed Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()