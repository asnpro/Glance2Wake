import cv2  # 导入OpenCV库，它是我们的"眼睛"

# 1. 打开默认摄像头 (0代表第一个摄像头)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头！请检查权限或设备连接。")
    exit()

print("摄像头已打开，按 'q' 键退出...")

while True:
    # 2. 一帧一帧地读取画面
    # ret是布尔值(成功/失败)，frame是画面数据
    ret, frame = cap.read()
    
    if not ret:
        print("无法接收画面")
        break

    # 3. 在窗口中显示画面
    cv2.imshow('Glance2Wake - Camera Test', frame)

    # 4. 监听键盘，如果按下 'q' 键，就退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 5. 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()