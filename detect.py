import pyrealsense2 as rs
import numpy as np
import cv2
import yaml

import torch
import torch.backends.cudnn as cudnn
from utils.general import check_img_size, non_max_suppression, set_logging, scale_coords
from utils.torch_utils import select_device, time_sync
from models.experimental import attempt_load
from utils.datasets import letterbox


# get camera pipline
pipeline = rs.pipeline()

# set configuration of the camera
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# start image stream
profile = pipeline.start(config)


# create an align object
align_to= rs.stream.color
align = rs.align(align_to)

# align_frames
def align_frames(frames):
      # align the depth frame to color frame
        aligned_frames = align.process(frames)

        #get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        RGB_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not RGB_frame:
            return 
        # transfrom RGB and depth image to ndarray
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        RGB_image = np.asanyarray(RGB_frame.get_data())

        return aligned_depth_frame, depth_image, RGB_image

class YOLOv5:
    def __init__(self, config_path = 'config\yolov5s.yaml'):
        with open(config_path, 'r', encoding= 'utf-8') as f:
          self.yolov5_config = yaml.load(f.read(), Loader= yaml.SafeLoader)
        self.colors = [[np.random.randint(0,255) for _ in range(3)] for _ in self.yolov5_config['class_name']]
        self.model_init()

    @torch.no_grad()
    def model_init(self):
        set_logging()
        self.device = select_device(self.yolov5_config['device'])
        self.half = self.device.type != 'cpu'  # CPU does not half precision
        self.model = attempt_load(weights= 'weights\yolov5s.pt', map_location= self.device)
        if self.half:
            self.model.half()
        # set cudnn.benchmark True can speed up constant image size inference
        cudnn.benchmark = True
        if self.device.type != 'cpu': # here the image size is only set to satify the demand of the network, it will be put into the network resized to image_size and will be returned as primitive size
            self.model(torch.zeros(1,3, self.yolov5_config['image_size'],self.yolov5_config['image_size']).to(self.device).type_as(next(self.model.parameters())))


    def plot_one_box(self, x, im, depth_img, color=(128, 128, 128), label=None, line_thickness=3):
        """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
        使用opencv在原图im上画一个bounding box
        :params x: 预测得到的bounding box  [x1 y1 x2 y2]
        :params im: 原图 要将bounding box画在这个图上  array
        :params color: bounding box线的颜色
        :params labels: 标签上的框框信息  类别 + score
        :params line_thickness: bounding box的线宽
        """
        # check im内存是否连续
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
        # tl = 框框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
        tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
        # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        # lower left
        lower_left =(int(x[0]), int(x[3]))
        # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
        # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # 如果label不为空还要在框框上面显示标签label + score
        if label:
            tf = max(tl - 1, 1)  # label字体的线宽 font thickness
            # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
            # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
            # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
            # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
            # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            # mid of the image
            mid_x, mid_y = (x[0]+x[2])//2, (x[1]+x[3])//2
            # distance
            distance =  depth_img.get_distance(mid_x, mid_y)
            # begin plot 
            text_size = t_size = cv2.getTextSize('distance', 0, fontScale=tl / 3, thickness=tf)[0]
            lower_left2 = lower_left[0] + text_size[0], lower_left[1] - text_size[1] - 3
            cv2.rectangle(im, lower_left, lower_left2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, 'distance: {}'.format(round(distance, 2)), (lower_left[0], lower_left[1] - 2),  0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return im
    
    @torch.no_grad()
    def detect(self, color_image, depth_image, view_results = True):
        img = color_image
        img = letterbox(img)[0]
        # turn BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # turn it to tensor
        self.torch_img = torch.from_numpy(img).to(self.device)
        self.torch_img = self.torch_img.half() if self.half else self.torch_img.float()  # uint8 to fp16/32
        self.torch_img = self.torch_img / 255.0
        self.torch_img = self.torch_img.unsqueeze(0) if self.torch_img.ndimension() == 3 else None

        # start inference
        start_time = time_sync()
        pred = self.model(self.torch_img, augment = False)[0]

        # NMS
        pred = non_max_suppression(pred, self.yolov5_config['threshold']['confidence'], self.yolov5_config['threshold']['iou'], classes =None, agnostic= False)
        end_time = time_sync()
        print('inference time: {}'.format(end_time - start_time))
        boxed_img = color_image
        for _, det in enumerate(pred):
                if len(det):
                     # scale_coords map the coordinates to primitive image
                    det[:, :4] = scale_coords(self.torch_img.shape[2:], det[:, :4], color_image.shape).round()
                    xyxy_list, conf_list, class_list = [], [], []
                    for *xyxy, conf, class_id in reversed(det):
                        xyxy_list.append(xyxy)
                        conf_list.append(conf)
                        class_list.append(int(class_id))
                        print('--'+self.yolov5_config['class_name'][int(class_id)])
                        if view_results:
                            label = '{}, {:.2f}'.format(self.yolov5_config['class_name'][int(class_id)], conf)
                            boxed_img = self.plot_one_box(xyxy, boxed_img, depth_image, color= self.colors[int(class_id)], label= label, line_thickness= 3 )
        return pred, det, boxed_img

if __name__ == '__main__':
    model = YOLOv5(config_path= 'config\yolov5s.yaml')
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('results.mp4',fourcc , fps, (640, 480) )
    count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth, depth_image, RGB_image = align_frames(frames= frames)
            if not depth_image.any() or not RGB_image.any():
                continue
            pred, det, boxed_img = model.detect(color_image= RGB_image, depth_image= depth)
            if count % 15 ==0:
                print('pred shape: {}, pred :{}'.format(pred[0].shape, pred))
                print('det shape: {}, det :{}'.format(det.shape, det)) 
                # print(reversed(det))
                # for *xyxy, conf, class_id in reversed(det):
                #     print(xyxy,'\n', conf,'\n', class_id)                  
            count +=1
            out.write(boxed_img)
            #show image
            cv2.namedWindow('RealSense Image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Image', boxed_img)
            key = cv2.waitKey(1)
            #press esp to quit
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        out.release()
        pipeline.stop()