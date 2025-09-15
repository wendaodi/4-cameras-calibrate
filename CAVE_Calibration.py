import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from pandas import read_table
import os

def reprojection(rot, trans, world_coor_3d, K, D):

    pts_reproj, jac = cv2.projectPoints(world_coor_3d, rot, trans, K, D)
    # reproj_xy = np.int32(pts_reproj).reshape(-1,2)
    reproj_xy = np.round(pts_reproj).reshape(-1,2)
    reproj_xy = np.int32(reproj_xy)

    return reproj_xy

def draw_points(img, img_pts, pts_id, color_flag, show_id=False, red=False):
    cirSize = 7
    textThickness = 2
    fontScale = 1
    color = (0, 0, 255)
    if color_flag == 'red':
        color = (0, 0, 255)
    elif color_flag == 'green':
        color = (0, 255, 0)
    elif color_flag == 'blue':
        color = (255, 0, 0)

    sz = img_pts.shape
    # img_pts_int = img_pts.astype(np.int32)
    img_pts_int = np.round(img_pts)
    img_pts_int = img_pts_int.astype(np.int32)
    for i in range(sz[1]):
        tmp_pts = (img_pts_int[0, i], img_pts_int[1, i])
        cv2.circle(img, tmp_pts, cirSize, color, -1)
        if(show_id):
            if(red):
                cv2.putText(img, "ID: " + str(pts_id[i] + 1), (int(img_pts_int[0, i] - 10), int(img_pts_int[1, i] - 10)), \
                    cv2.FONT_HERSHEY_DUPLEX  , fontScale, color, textThickness, cv2.LINE_AA)
            else:
                cv2.putText(img, "ID: " + str(pts_id[i] + 1), (int(img_pts_int[0, i] + 10), int(img_pts_int[1, i] + 10)), \
                    cv2.FONT_HERSHEY_DUPLEX  , fontScale, color, textThickness, cv2.LINE_AA)

    return img
if __name__ == "__main__":
    
    camera_num = 3
    points_file = './CAVE_img/{}.json'.format(camera_num)
    img_path = './CAVE_img/{}/0.png'.format(camera_num)
    world_pts_path = './CAVE_calibration/Data/CAVE2.txt'
    param_save = './CAVE_img'

    # img_path = 'D:/taiic/CAVE32/CAVE_Calib/CAVE_img/1/0.png'.format(camera_num)
    
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = img.shape

    with open(points_file) as f:
        point_file = json.load(f)

    world_data = []
    with open(world_pts_path) as file:
        file_data = file.readlines()
        for row in file_data:
            data_line = row.split(',')[1:4] #按‘，’切分每行的数据
            world_data.append(data_line) #将每行数据插入data中
    data = np.asarray(world_data)
    world_points_all = data.astype(np.float32)

    # i = np.zeros((5, 3))
    # i[2,1] = 1
    # i_ = []
    # i_.append(i)

    points_lable = point_file['shapes']
    points_id = []
    img_points = []
    img_pts = []
    for i in range(len(points_lable)):
        point_id = points_lable[i]['label']
        point = points_lable[i]['points']
        point = (np.asarray(point)).squeeze()
        points_id.append(int(point_id)-1)
        img_points.append(point)

    # img_pts = np.stack(img_points, axis=0)
    img_pts.append(np.asarray(img_points, dtype=np.float32))
    # img_points = np.asarray(img_points)
    # img_points = np.array([[corner for [corner] in img_points]])

    world_pts = []
    world_points = np.flip(world_points_all, axis=0)
    world_points = world_points[2:, :]#只保留棋盘格的点
    world_points = world_points[points_id]
    world_pts.append(world_points)

    # camera_matrix = cv2.initCameraMatrix2D(world_pts, img_pts, img_gray.shape[::-1])
    # camera_matrix = np.array([[617.51, 0, 639.29],[0, 619.56, 373.2],[0, 0, 1]],dtype=np.float32)
    camera_matrix = np.array([[610.15199034, 0.0, 646.16469872],[0.0, 609.89054978, 374.4133134],[0, 0, 1]],dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_pts, img_pts, img_gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    rvecs = np.asarray(rvecs).squeeze(0)
    tvecs = np.asarray(tvecs).squeeze(0)
    rvecs, jacb = cv2.Rodrigues(rvecs)
    Rt = np.hstack((rvecs,tvecs))
    with open(os.path.join(param_save, "K{}.txt".format(camera_num)), 'w') as f:
        np.savetxt(f, mtx, fmt='%.08f')
    with open(os.path.join(param_save, "Rt{}.txt".format(camera_num)), 'w') as f1:
        np.savetxt(f1, Rt, fmt='%.08f')
    # np.savetxt(param_save + "t{}.txt".format(camera_num), tvecs)
    print(f'K:\n{mtx}\n R:\n{rvecs}\n t:\n{tvecs}')
    print(dist)

    # reprojection
    # cam_coor_3d = np.matmul(rot, world_points) + tvecs
    # pts_reproj, jac = cv2.projectPoints(world_points, Rot, tvecs, mtx, dist)
    # reproj_xy = np.int32(pts_reproj).reshape(-1,2)

    img_pts = np.stack(img_points, axis=1)
    world_points = world_points.transpose()

    pts_reproj = reprojection(rvecs, tvecs, world_points, mtx, dist)
    pts_reproj = pts_reproj.transpose()
    truth_imgpoints = np.asarray(img_points, dtype=np.float32).transpose()
    reproj_error = np.mean(np.linalg.norm(truth_imgpoints - pts_reproj, axis=0))
    print(f'mean of reproject error:{reproj_error}\n')
    img = draw_points(img, pts_reproj, points_id, 'red', show_id=True, red=True)
    img = draw_points(img, img_pts, points_id, 'green',show_id=True)

    # 投影三个轴上的点
    a_pts = np.arange(0, 4.0, 0.1)
    axis_x = np.zeros((3,a_pts.shape[0]), dtype=np.float32)
    axis_x[0,:] = a_pts
    repro_x = reprojection(rvecs, tvecs, axis_x, mtx, dist)
    pts_id = np.arange(world_points.shape[1], world_points.shape[1] + a_pts.shape[0])
    img = draw_points(img, repro_x.T, pts_id, 'red')

    axis_y = np.zeros((3,a_pts.shape[0]), dtype=np.float32)
    axis_y[1,:] = a_pts
    repro_y = reprojection(rvecs, tvecs, axis_y, mtx, dist)
    img = draw_points(img, repro_y.T, pts_id, 'green')

    axis_z = np.zeros((3,a_pts.shape[0]), dtype=np.float32)
    axis_z[2,:] = a_pts
    repro_z = reprojection(rvecs, tvecs, axis_z, mtx, dist)
    img = draw_points(img, repro_z.T, pts_id, 'blue')

    # print(pts_reproj)

    # cv2.imshow(windowName, img)
    # cv2.imwrite("./CAVE_img/test.png", img)
    cv2.imwrite("./CAVE_img/{}/reproj_{}.png".format(camera_num, camera_num), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()