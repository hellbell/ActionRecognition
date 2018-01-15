import os
import cv2

data_root = '/home/sangdoo/work/dataset/UCF11_updated_mpg'
save_root = '/home/sangdoo/work/dataset/UCF11'

root_dir = os.listdir(data_root)
root_dir = sorted(root_dir)

for rd in root_dir:
    rd_path = os.path.join(data_root, rd)

    cls_dir = os.listdir(rd_path)
    cls_dir = sorted(cls_dir)

    for cd in cls_dir:
        if cd == 'Annotation':
            continue
        cd_path = os.path.join(rd_path, cd)

        vid_dir = os.listdir(cd_path)
        vid_dir = sorted(vid_dir)
        for vd in vid_dir:
            vd_path = os.path.join(cd_path, vd)
            vd_name = vd[:-4]
            save_path = os.path.join(save_root, rd, vd_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # load video from vd_path
            vidcap = cv2.VideoCapture(vd_path)

            # split video
            success, image = vidcap.read()
            count = 0
            success = True
            while success:
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                cv2.imwrite(os.path.join(save_path, "%05d.png" % count), image)  # save frame as JPEG file
                count += 1

            # save video frames into save_path

            idx = 0
