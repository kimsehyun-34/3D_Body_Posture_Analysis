import os
import cv2
import numpy as np
from PIL import Image
import glob
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512), output_format='png'):
        """
        이미지 전처리 클래스
        
        Args:
            target_size: 목표 이미지 크기 (width, height)
            output_format: 출력 이미지 포맷 ('png', 'jpg', 'bmp')
        """
        self.target_size = target_size
        self.output_format = output_format.lower()
        
    def preprocess_image(self, image_path, output_path):
        """
        단일 이미지 전처리
        
        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로
        """
        try:
            # PIL로 먼저 시도
            try:
                with Image.open(image_path) as pil_image:
                    # PIL 이미지를 numpy 배열로 변환
                    image = np.array(pil_image)
                    
                    # RGB를 BGR로 변환 (OpenCV 형식)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    print(f"PIL로 로드 성공: {image_path}")
            except Exception:
                # PIL 실패시 OpenCV로 시도
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Error: 이미지를 읽을 수 없습니다 - {image_path}")
                    return False
                print(f"OpenCV로 로드 성공: {image_path}")
            
            # 원본 이미지 정보
            original_height, original_width = image.shape[:2]
            print(f"원본 크기: {original_width} x {original_height}")
            
            # 이미지 리사이즈 (비율 유지하면서)
            resized_image = self.resize_with_aspect_ratio(image, self.target_size)
            
            # 정사각형으로 패딩 (필요한 경우)
            final_image = self.add_padding_to_square(resized_image, self.target_size)
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 이미지 저장 (PIL 사용)
            try:
                if len(final_image.shape) == 3:
                    # BGR을 RGB로 변환
                    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                    pil_output = Image.fromarray(rgb_image)
                else:
                    # 그레이스케일
                    pil_output = Image.fromarray(final_image)
                
                # PIL로 저장
                pil_output.save(output_path)
                print(f"전처리 완료: {output_path}")
                return True
                
            except Exception as save_error:
                print(f"PIL 저장 실패, OpenCV로 재시도: {save_error}")
                # 백업으로 OpenCV 사용
                success = False
                if self.output_format == 'png':
                    success = cv2.imwrite(output_path, final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                elif self.output_format == 'jpg':
                    success = cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    success = cv2.imwrite(output_path, final_image)
                
                if success:
                    print(f"전처리 완료: {output_path}")
                    return True
                else:
                    print(f"Error: 이미지 저장 실패 - {output_path}")
                    return False
            
        except Exception as e:
            print(f"Error 이미지 전처리 중 오류: {e}")
            return False
    
    def resize_with_aspect_ratio(self, image, target_size):
        """
        비율을 유지하면서 이미지 리사이즈
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # 비율 계산
        ratio = min(target_width / width, target_height / height)
        
        # 새로운 크기 계산
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 리사이즈
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def add_padding_to_square(self, image, target_size):
        """
        이미지를 정사각형으로 만들기 위해 패딩 추가
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # 패딩 계산
        pad_width = (target_width - width) // 2
        pad_height = (target_height - height) // 2
        
        # 패딩 추가 (검은색으로)
        if len(image.shape) == 3:  # 컬러 이미지
            padded = cv2.copyMakeBorder(
                image, 
                pad_height, target_height - height - pad_height,
                pad_width, target_width - width - pad_width,
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
        else:  # 그레이스케일 이미지
            padded = cv2.copyMakeBorder(
                image, 
                pad_height, target_height - height - pad_height,
                pad_width, target_width - width - pad_width,
                cv2.BORDER_CONSTANT, 
                value=0
            )
        
        return padded
    
    def preprocess_folder(self, input_folder, output_folder):
        """
        폴더 내 모든 이미지 전처리
        
        Args:
            input_folder: 입력 폴더 경로
            output_folder: 출력 폴더 경로
        """
        # 지원하는 이미지 확장자
        supported_extensions = ['*.bmp', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        print(f"발견된 이미지 파일: {len(image_files)}개")
        
        success_count = 0
        for image_path in image_files:
            # 출력 파일명 생성
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            output_filename = f"{name}.{self.output_format}"
            output_path = os.path.join(output_folder, output_filename)
            
            # 이미지 전처리
            if self.preprocess_image(image_path, output_path):
                success_count += 1
        
        print(f"전처리 완료: {success_count}/{len(image_files)}개 파일")
        return success_count, len(image_files)
    
    def preprocess_all_subfolders(self, root_folder, output_root):
        """
        루트 폴더의 모든 하위 폴더들을 재귀적으로 전처리
        
        Args:
            root_folder: 입력 루트 폴더
            output_root: 출력 루트 폴더
        """
        total_success = 0
        total_files = 0
        
        for root, dirs, files in os.walk(root_folder):
            # 이미지 파일이 있는 폴더만 처리
            image_files = [f for f in files if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
            
            if image_files:
                # 상대 경로 계산
                rel_path = os.path.relpath(root, root_folder)
                output_folder = os.path.join(output_root, rel_path)
                
                print(f"\n처리 중인 폴더: {root}")
                print(f"출력 폴더: {output_folder}")
                
                success, total = self.preprocess_folder(root, output_folder)
                total_success += success
                total_files += total
        
        print(f"\n전체 전처리 완료: {total_success}/{total_files}개 파일")
        return total_success, total_files

def test_single_folder():
    """
    테스트 함수 - 단일 폴더만 전처리
    """
    # 경로 설정
    input_folder = r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\data\DepthMap\오른쪽_남(비정상)"
    output_folder = r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\data\Test_Preprocessed"
    
    # 전처리 설정
    target_size = (512, 512)  # 512x512 픽셀로 통일
    output_format = 'png'     # PNG 포맷으로 저장
    
    print("=== 테스트 이미지 전처리 시작 ===")
    print(f"입력 폴더: {input_folder}")
    print(f"출력 폴더: {output_folder}")
    print(f"목표 크기: {target_size}")
    print(f"출력 포맷: {output_format}")
    print("=" * 50)
    
    # 전처리 객체 생성
    preprocessor = ImagePreprocessor(target_size=target_size, output_format=output_format)
    
    # 폴더 전처리 (처음 5개만)
    success, total = preprocessor.preprocess_folder(input_folder, output_folder)
    
    print("\n=== 테스트 완료 ===")
    print(f"총 처리된 파일: {success}/{total}")
    print(f"성공률: {(success/total*100):.1f}%" if total > 0 else "파일 없음")
    
    return success > 0

def main():
    """
    메인 함수 - 데이터 폴더의 모든 이미지 전처리
    """
    # 먼저 테스트 실행
    print("테스트 모드로 실행 중...")
    test_result = test_single_folder()
    
    if not test_result:
        print("테스트 실패. 전체 처리를 중단합니다.")
        return
    
    print("\n테스트 성공! 전체 처리를 시작합니다...")
    
    # 경로 설정
    input_root = r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\data\DepthMap"
    output_root = r"d:\기타\파일 자료\파일\프로젝트 PJ\3D_Body_Posture_Analysis\data\Preprocessed_DepthMap"
    
    # 전처리 설정
    target_size = (512, 512)  # 512x512 픽셀로 통일
    output_format = 'png'     # PNG 포맷으로 저장
    
    print("\n=== 전체 이미지 전처리 시작 ===")
    print(f"입력 폴더: {input_root}")
    print(f"출력 폴더: {output_root}")
    print(f"목표 크기: {target_size}")
    print(f"출력 포맷: {output_format}")
    print("=" * 50)
    
    # 전처리 객체 생성
    preprocessor = ImagePreprocessor(target_size=target_size, output_format=output_format)
    
    # 모든 하위 폴더 전처리
    success, total = preprocessor.preprocess_all_subfolders(input_root, output_root)
    
    print("\n=== 전처리 완료 ===")
    print(f"총 처리된 파일: {success}/{total}")
    print(f"성공률: {(success/total*100):.1f}%" if total > 0 else "파일 없음")

if __name__ == "__main__":
    main()
