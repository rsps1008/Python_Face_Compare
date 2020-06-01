import sys, os, dlib, glob, numpy
from skimage import io
import cv2
import imutils

if len(sys.argv) != 2:
    print("請輸入要辨識的圖片檔名")
    exit()

# 依據臉部68個特徵預測臉型, 臉面偵測模型
path_face_landmarks_pattern = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型
path_face_recognition_pattern = "dlib_face_recognition_resnet_model_v1.dat"

# 人臉資料庫路徑
path_face_database = "./facebase"

# 要辨識的圖片檔名
path_img = sys.argv[1]

# 載入臉型檢測器
detector_face_shape = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
detector_face_landmarks = dlib.shape_predictor(path_face_landmarks_pattern)

# 載入人臉辨識檢測器
detector_face_recognition= dlib.face_recognition_model_v1(path_face_recognition_pattern)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
for f in glob.glob(os.path.join(path_face_database, "*.jpg")):
    osPath = os.path.basename(f) #去除路徑, 只顯示檔案名稱
    candidate.append(os.path.splitext(osPath)[0])#splittext將檔名及副檔名切開
    img = io.imread(f) #將資料庫裏的圖片一張一張載入
    #人臉偵測
    face_shape = detector_face_shape(img, 1)

    for k, d in enumerate(face_shape):
        #特徵點偵測
        face_landmarks = detector_face_landmarks(img, d)

        #取得描述子，128維特徵向量
        face_descriptor = detector_face_recognition.compute_face_descriptor(img, face_landmarks)

        #轉換numpy array格式
        v = numpy.array(face_descriptor)
        descriptors.append(v)

img = io.imread(path_img)
face_shape = detector_face_shape(img, 1)
dist = []
for k, d in enumerate(face_shape):
    #k是第幾組的意思, 由0開始, d是臉型在圖片的左上角及右下角, 如 d: [(153,67) (282,196)]
    face_landmarks = detector_face_landmarks(img, d)
    face_descriptor = detector_face_recognition.compute_face_descriptor(img, face_landmarks)
    d_test = numpy.array(face_descriptor)

    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    # 計算歐式距離
    for i in descriptors:
        dist_ = numpy.linalg.norm(i - d_test)
        dist.append(dist_)

# 將比對人名和比對出來的歐式距離組成一個dict
c_d = dict(zip(candidate, dist))

# 根據歐式距離由小到大排序
cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
# 取得最短距離就為辨識出的人名
rec_name = cd_sorted[0][0]

# 將辨識出的人名印到圖片上面
print(rec_name);
cv2.putText(img, rec_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 4, (100, 255, 255), 8, cv2.LINE_AA)

img = imutils.resize(img, width=600)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Face Recognition", img)
# 隨意Key一鍵結束程式
cv2.waitKey(0)
cv2.destroyAllWindows()