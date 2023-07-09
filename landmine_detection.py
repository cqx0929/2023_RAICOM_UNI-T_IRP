import os
import cv2
import numpy as np
import time
import concurrent.futures


def process_image(img_load_path, img_save_path, p_name):
    try:

        # 读取图片
        img = cv2.imread(img_load_path, cv2.IMREAD_GRAYSCALE)

        # 图片预处理，调整大小，转成灰度图
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))

        # 二值化，去除灰色块的影响
        ret, threshold_img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

        '''腐蚀掉其他部分保留黑色方块，以此进行旋转'''

        # 定义腐蚀和膨胀的内核
        kernel = np.ones((10, 10), np.uint8)

        # 腐蚀掉黑色方块以外的区域
        dilated_img = cv2.dilate(threshold_img, kernel, iterations=1)

        # 将黑色方块膨胀至接近原来的大小
        eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
        _, threshold_img0 = cv2.threshold(eroded_img, 90, 255, cv2.THRESH_BINARY)

        # 获取黑色方块位置，边缘，中心点
        bs_ys, bs_xs = np.where(threshold_img0 == 0)
        bs_edge = [min(bs_xs), max(bs_xs), min(bs_ys), max(bs_ys)]  # 黑方块的边界 left right bottom top
        bs_center = [int((bs_edge[0] + bs_edge[1]) / 2), int((bs_edge[2] + bs_edge[3]) / 2)]  # x, y

        # 调整坐标，使其相对于(0, 0)
        bs_center = (bs_center[0] - int(img.shape[1] / 2), int(img.shape[0] / 2) - bs_center[1])  # x-w/2, h/2-y

        # 旋转
        rotate_img = threshold_img
        if bs_center[0] > 0 and bs_center[1] > 0:  # x>0 y>0第 一 象限，0
            pass
        elif bs_center[0] < 0 < bs_center[1]:  # x<0 y>0 第 二 象限，顺时针旋转90
            rotate_img = cv2.rotate(threshold_img, cv2.ROTATE_90_CLOCKWISE)
        elif bs_center[1] < 0 < bs_center[0]:  # x>0 y<0 第 三 象限，顺时针旋转180
            rotate_img = cv2.rotate(threshold_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif bs_center[0] < 0 and bs_center[1] < 0:  # x<0 y<0 第 四 象限，逆时针90
            rotate_img = cv2.rotate(threshold_img, cv2.ROTATE_180)
        else:
            pass  # 坐标轴上或坐标原点，几乎不可能

        '''寻找四边形并进行投影裁切'''

        # 膨胀的内核
        kernel = np.ones((2, 2), np.uint8)

        # 膨胀将黑矩形加粗
        eroded_img = cv2.erode(rotate_img, kernel, iterations=2)

        # 提取边缘
        canny_img = cv2.Canny(eroded_img, 10, 100)

        # 轮廓检测
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寻找最大四边形轮廓
        max_area = -1
        max_quad = None
        for contour in contours:

            # 计算周长
            perimeter = cv2.arcLength(contour, True)

            # 对轮廓进行多边形逼近，指定逼近精度为周长的4 %
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            # 检查逼近的多边形是否具有4个顶点
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    max_quad = approx
                else:
                    pass
            else:
                pass

        # 排序顶点
        h, w = rotate_img.shape
        points = tuple(max_quad.squeeze().tolist())
        tl = points[0]
        tr = points[0]
        bl = points[0]
        br = points[0]
        for point in points:
            x, y = point[0], point[1]
            if w - x + y < w - tr[0] + tr[1]:
                tr = (x, y)
            if x + h - y < bl[0] + h - bl[1]:
                bl = (x, y)
            if x + y < tl[0] + tl[1]:
                tl = (x, y)
            if w - x + h - y < w - br[0] + h - br[1]:
                br = (x, y)

        # 定义原始尺寸和目标尺寸
        src_pts = np.array([tl, tr, bl, br], dtype=np.float32)
        dst_width = 840
        dst_height = 540
        dst_pts = np.array([[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]], dtype=np.float32)

        # 计算变换矩阵
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 投影裁切
        warp_perspective_img = cv2.warpPerspective(rotate_img, perspective_matrix, (dst_width, dst_height))

        '''霍夫检测找圆并进行区域划分输出结果'''

        # 找圆
        circles = cv2.HoughCircles(
            warp_perspective_img,  # 图片
            method=cv2.HOUGH_GRADIENT,  # 霍夫方法
            dp=1,  # 图像解析的反向比例。1为原始大小，2为原始大小的一半
            minDist=35,  # 圆心之间的最小距离。过小会增加圆的误判，过大会丢失存在的圆
            param1=90,  # Canny检测器的高阈值
            param2=15,  # 检测阶段圆心的累加器阈值。越小的话，会增加不存在的圆；越大的话，则检测到的圆就更加接近完美的圆形
            minRadius=20,  # 检测的最小圆的半径
            maxRadius=40  # 检测的最大圆的半径
        )

        # 去黑边 h, w
        warp_perspective_img[:20, :] = 255  # 上
        warp_perspective_img[dst_height - 20:, :] = 255  # 下
        warp_perspective_img[:, :20] = 255  # 左
        warp_perspective_img[:, dst_width - 20:] = 255  # 右

        # 加性感黑边
        warp_perspective_img[:5, :] = 0  # 上
        warp_perspective_img[dst_height - 5:, :] = 0  # 下
        warp_perspective_img[:, :5] = 0  # 左
        warp_perspective_img[:, dst_width - 5:] = 0  # 右

        # 转换色彩空间，方便绘制结果
        bgr_img = cv2.cvtColor(warp_perspective_img, cv2.COLOR_GRAY2BGR)

        # 四舍五入, 然后转为整数
        circles = np.uint16(np.around(circles))[0]

        # 分割区域
        x_mark = 'ABCDEF'
        y_mark = '1234'
        positions = []

        # 遍历每个圆
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]

            # 标记区域
            circle_position = \
                x_mark[int(center[0] / dst_width * 6 // 1)] + \
                y_mark[int((center[1]) / dst_height * 4 // 1)]
            positions += [circle_position]
            cv2.circle(
                bgr_img,
                center=center,
                radius=radius,
                color=(0, 255, 0),
                thickness=3
            )
            cv2.rectangle(
                bgr_img,
                pt1=(center[0] - radius, center[1] - radius),
                pt2=(center[0] + radius, center[1] + radius),
                color=(0, 255, 0),
                thickness=3
            )

        '''格式化结果并输出'''

        # 排序
        positions = sorted(positions, key=lambda i: (i[0], i[1]))
        position_sorted = ''
        for position in positions:
            position_sorted += f' {position}'

        # 保存图片
        cv2.imwrite(img_save_path, bgr_img)

        '''返回结果'''

        return [bgr_img, position_sorted, p_name]

    except Exception as e:
        print(f'图片处理出错：{p_name}，错误信息：{str(e)}')
        return None


def run(
        img_load_dir='smart_detect_dataset/P1',  # 图片读取文件夹
        res_save_dir='run/res',  # 结果保存文件夹
        show_img=False,
        num_threads=128
):
    try:
        process_start = time.perf_counter()  # 程序计时起点

        # 创建输出文件夹
        os.makedirs(res_save_dir, exist_ok=True)

        # 删除结果目录中所有文件
        res_files = os.listdir(res_save_dir)
        if res_files:
            for file in res_files:
                res_files_path = os.path.join(res_save_dir, file)
                os.remove(res_files_path)

        # 检测结果保存路径
        txt_save_path = os.path.join(res_save_dir, 'detect_info.txt')

        # 图片路径
        p_names = os.listdir(img_load_dir)
        img_load_paths = [os.path.join(img_load_dir, p_name) for p_name in p_names]
        img_save_paths = [os.path.join(res_save_dir, 'detected_'+p_name) for p_name in p_names]

        # 多线程读取处理图片
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            res_list = executor.map(process_image, img_load_paths, img_save_paths, p_names)

        # 程序性能分析
        process_time = time.perf_counter() - process_start  # 程序运行总时间
        print(f'process time:{process_time:.2f}s')
        img_detect_num_per_second = len(p_names) / process_time  # 检测速率
        print(f'rate:{img_detect_num_per_second:.2f}s')

        # 图片展示
        s = ''
        for bgr_img, position_sorted, p_name in res_list:

            # 地雷检测信息收集并输出
            positions_info = f'{p_name}地雷所在区域:{position_sorted}'
            print(positions_info)
            s += f'{positions_info}\n'

            # 图片展示
            if show_img:
                cv2.imshow(p_name + position_sorted, bgr_img)
                if cv2.waitKey(0):  # 按任意键继续
                    pass
                cv2.destroyAllWindows()

        # 保存检测结果文档
        with open(txt_save_path, 'w+', encoding='utf-8') as fp:
            fp.write(s)

    except Exception as e:
        print(f'发生错误{str(e)}！')


if __name__ == '__main__':
    run()
