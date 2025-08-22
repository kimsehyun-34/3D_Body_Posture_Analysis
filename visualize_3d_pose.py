import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def load_depth_map(file_path):
    # DepthMap 이미지를 로드하고 깊이 정보로 변환
    depth_map = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return depth_map

def create_3d_points(depth_map, view):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    if view == "front":
        return x, y, depth_map
    elif view == "right":
        return depth_map, y, width - x
    elif view == "left":
        return -depth_map, y, x
    elif view == "back":
        return x, y, -depth_map
    
    return None

def visualize_3d_pose():
    # 각 뷰의 DepthMap 로드
    front_depth = load_depth_map(r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\정면_남\DepthMap0.bmp")
    right_depth = load_depth_map(r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\오른쪽_남\DepthMap0.bmp")
    left_depth = load_depth_map(r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\왼쪽_남\DepthMap0.bmp")
    back_depth = load_depth_map(r"d:\기타\파일 자료\파일\프로젝트 PJ\AAAAAA2_3D_자세\test\정상\후면_남\DepthMap0.bmp")

    # 3D 포인트 생성
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # 각 뷰의 3D 포인트 시각화
    for depth_map, view, c in [(front_depth, "front", 'red'),
                              (right_depth, "right", 'blue'),
                              (left_depth, "left", 'green'),
                              (back_depth, "back", 'yellow')]:
        
        # 포인트 수를 줄이기 위해 다운샘플링
        step = 10
        x, y, z = create_3d_points(depth_map[::step, ::step], view)
        
        # 깊이값이 0인 점들은 제외
        mask = z > 0
        ax.scatter(x[mask], y[mask], z[mask], c=c, alpha=0.1, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('3D Pose Visualization')
    plt.show()

if __name__ == "__main__":
    visualize_3d_pose()
