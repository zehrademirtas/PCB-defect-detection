import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def load_image(image_path):
    """Görüntüyü belirtilen yoldan yükler."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")
    return image

def image_registration(reference_image, test_image):
    """Referans ve test görüntülerini çakıştırır."""
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(reference_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(test_image, None)
    
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors not found, check the images for proper content.")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found between the images.")

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    height, width, channels = reference_image.shape
    aligned_image = cv2.warpPerspective(test_image, matrix, (width, height))
    
    return aligned_image

def find_differences(reference_image, test_image):
    """İki görüntü arasındaki farkları tespit eder."""
    diff = cv2.absdiff(reference_image, test_image)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return thresh

def parse_annotation(xml_path):
    """XML dosyasını parçalayıp anotasyonları çıkarır."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

def draw_bounding_boxes(image, bboxes, color):
    """Görüntü üzerine bounding box'ları çizer."""
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

def main():
    # Veri seti yolları
    dataset_path = r"C:\Users\zehra\OneDrive\Desktop\23010903135_22010903087"
    reference_path = os.path.join(dataset_path, "Reference", "01.JPG")
    rotation_path = os.path.join(dataset_path, "rotation")
    annotation_path = os.path.join(dataset_path, "Annotations")
    
    # Referans görüntüyü yükleme
    try:
        reference_image = load_image(reference_path)
    except Exception as e:
        print(f"Referans görüntüsü yüklenemedi: {e}")
        return
    
    defect_types = ["Missing_hole", "Mouse_bite", "Open_circuit"]
    
    for defect_type in defect_types:
        rotation_folder = os.path.join(rotation_path, defect_type + "_rotation")
        annotation_folder = os.path.join(annotation_path, defect_type)
        
        for i in range(1, 21):
            try:
                test_image_path = os.path.join(rotation_folder, f"01_{defect_type.lower()}_{i:02}.JPG")
                annotation_file_path = os.path.join(annotation_folder, f"01_{defect_type.lower()}_{i:02}.xml")
                
                # Eksik dosyaları sessizce atla
                if not os.path.exists(test_image_path) or not os.path.exists(annotation_file_path):
                    continue
                
                # Test görüntüsünü yükleme
                test_image = load_image(test_image_path)
                
                # Görüntüleri çakıştırma
                aligned_image = image_registration(reference_image, test_image)
                
                # Farkları bulma
                differences = find_differences(reference_image, aligned_image)
                
                # Anotasyonları yükleme
                bboxes = parse_annotation(annotation_file_path)
                
                # Tespit edilen kusurları çerçeve içine alma
                defect_image = reference_image.copy()
                draw_bounding_boxes(defect_image, bboxes, (0, 255, 0))
                
                # Tespit edilen farkları çerçeve içine alma
                contours, _ = cv2.findContours(differences, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(defect_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Sonuç görüntüsünü kaydetme
                result_dir = os.path.join(dataset_path, "Results")
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                result_path = os.path.join(result_dir, f"{defect_type}_{i:02}.JPG")
                cv2.imwrite(result_path, defect_image)
                print(f"Sonuç kaydedildi: {os.path.abspath(result_path)}")
            
            except Exception as e:
                continue  # Bir sonraki görüntüye geç

if __name__ == "__main__":
    main()
