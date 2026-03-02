import cv2
import numpy as np
import kagglehub
import os
import pickle
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


IMG_SIZE    = 64
MAX_IMAGES  = 200
RANDOM_SEED = 42
MODEL_FILE  = 'gesture_model.pkl'  

DATASET_PATH = r'C:\Users\USER\.cache\kagglehub\datasets\gti-upm\leapgestrecog\versions\1\leapGestRecog'


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    return features


def train_model():
    print('📂 Loading dataset...')
    images, labels = [], []
    class_counts   = {}
    subjects       = sorted(os.listdir(DATASET_PATH))

    for subject in subjects:
        subject_path = os.path.join(DATASET_PATH, subject)
        if not os.path.isdir(subject_path):
            continue
        for gesture_folder in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue
            gesture_name = '_'.join(gesture_folder.split('_')[1:])
            if gesture_name not in class_counts:
                class_counts[gesture_name] = 0
            if class_counts[gesture_name] >= MAX_IMAGES:
                continue
            for fname in os.listdir(gesture_path):
                if class_counts[gesture_name] >= MAX_IMAGES:
                    break
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img = cv2.imread(os.path.join(gesture_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
                labels.append(gesture_name)
                class_counts[gesture_name] += 1

    print(f'✅ Loaded {len(images)} images across {len(class_counts)} classes')

    print('⚙️  Extracting HOG features...')
    X = np.array([extract_features(img) for img in images])

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print('🤖 Training SVM model...')
    model = SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=RANDOM_SEED)
    model.fit(X_train_sc, y_train)

    acc = model.score(X_test_sc, y_test)
    print(f'✅ Model trained! Accuracy: {acc*100:.2f}%')

   
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'le': le}, f)
    print(f'💾 Model saved to {MODEL_FILE}')

    return model, scaler, le

def load_or_train():
    if os.path.exists(MODEL_FILE):
        print(f'✅ Loading saved model from {MODEL_FILE}...')
        with open(MODEL_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler'], data['le']
    else:
        return train_model()


def run_camera(model, scaler, le):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('❌ Cannot open webcam!')
        return

    print('\n✅ Webcam opened!')
    print('📌 Instructions:')
    print('   - Show your hand gesture in the GREEN BOX')
    print('   - Press Q to quit')
    print('   - Press S to save a screenshot\n')

   
    GREEN      = (0, 255, 0)
    WHITE      = (255, 255, 255)
    BLACK      = (0, 0, 0)
    YELLOW     = (0, 255, 255)
    RED        = (0, 0, 255)
    DARK_BG    = (30, 30, 30)

    gesture_history = []   
    HISTORY_SIZE    = 10

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('❌ Failed to read frame')
            break

        frame = cv2.flip(frame, 1)   
        h, w  = frame.shape[:2]

        
        roi_x1, roi_y1 = w//2 - 120, h//2 - 120
        roi_x2, roi_y2 = w//2 + 120, h//2 + 120

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        
        features    = extract_features(roi).reshape(1, -1)
        features_sc = scaler.transform(features)
        proba       = model.predict_proba(features_sc)[0]
        pred_idx    = np.argmax(proba)
        confidence  = proba[pred_idx] * 100
        gesture     = le.classes_[pred_idx].upper()

       
        gesture_history.append(gesture)
        if len(gesture_history) > HISTORY_SIZE:
            gesture_history.pop(0)
       
        stable_gesture = max(set(gesture_history), key=gesture_history.count)

        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, h), DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

       
        box_color = GREEN if confidence > 70 else YELLOW
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, 3)
        cv2.putText(frame, 'PLACE HAND HERE', (roi_x1, roi_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        
        cv2.putText(frame, 'GESTURE', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.putText(frame, 'DETECTOR', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        cv2.line(frame, (10, 80), (210, 80), GREEN, 1)

        
        cv2.putText(frame, 'DETECTED:', (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, stable_gesture, (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3)

        
        cv2.putText(frame, f'Confidence:', (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        bar_w = int(200 * confidence / 100)
        bar_color = GREEN if confidence > 70 else (YELLOW if confidence > 40 else RED)
        cv2.rectangle(frame, (10, 195), (210, 215), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, 195), (10 + bar_w, 215), bar_color, -1)
        cv2.putText(frame, f'{confidence:.1f}%', (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        
        cv2.line(frame, (10, 245), (210, 245), (80, 80, 80), 1)
        cv2.putText(frame, 'TOP PREDICTIONS:', (10, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        top3_idx = np.argsort(proba)[::-1][:3]
        for i, idx in enumerate(top3_idx):
            g_name = le.classes_[idx].upper()
            g_prob = proba[idx] * 100
            color  = GREEN if i == 0 else (180, 180, 180)
            cv2.putText(frame, f'{i+1}. {g_name}: {g_prob:.1f}%',
                        (10, 290 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        
        cv2.rectangle(frame, (0, h-35), (w, h), DARK_BG, -1)
        cv2.putText(frame, 'Q: Quit  |  S: Screenshot', (w//2 - 120, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('✋ Hand Gesture Recognition — Prodigy Infotech Task 04', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print('👋 Exiting...')
            break
        elif key == ord('s') or key == ord('S'):
            screenshot_count += 1
            fname = f'screenshot_{screenshot_count}.png'
            cv2.imwrite(fname, frame)
            print(f'📸 Screenshot saved: {fname}')

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print('=' * 50)
    print('  ✋ Hand Gesture Recognition')
    print('  Prodigy Infotech — Task 04')
    print('=' * 50)

    model, scaler, le = load_or_train()
    run_camera(model, scaler, le)
