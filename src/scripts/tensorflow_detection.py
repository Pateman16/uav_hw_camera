#!/usr/bin/env python
# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

#Tensorflow imports
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import math
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream, SessionWorker

import time
import sys
sys.path.insert(0, '/home/nvidia/catkin_ws/src/uav_hw_camera/src/scripts/')

## LOAD CONFIG PARAMS ##
if (os.path.isfile('/home/nvidia/catkin_ws/src/uav_hw_camera/src/scripts/config.yml')):
    with open("/home/nvidia/catkin_ws/src/uav_hw_camera/src/scripts/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    with open("config.sample.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
max_frames          = cfg['max_frames']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_interval        = cfg['det_interval']
det_th              = cfg['det_th']
model_name          = cfg['model_name']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
split_model         = cfg['split_model']
log_device          = cfg['log_device']
ssd_shape           = cfg['ssd_shape']

class tensorflow_detection:
    def __init__(self):

        self.detection_graph, score, expand = self.load_frozenmodel()
        self.category_index = self.load_labelmap()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
        config.gpu_options.allow_growth = allow_memory_growth
        cur_frames = 0
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:
                self.cam_read = Image()
                # topic where we publish
                self.image_pub = rospy.Publisher("/tensorflow_detection/image", Image, queue_size=10)
                self.bridge = CvBridge()
                # topic where the coordinates go
                self.cam_pose_pub = rospy.Publisher("/tensorflow_detection/cam_point", Point, queue_size=1)
                self.cam_pose = Point()
                self.cam_pose_center_pub = rospy.Publisher("/tensorflow_detection/cam_point_center", Point, queue_size=1)
                self.cam_pose_center = Point()
                # subscribed Topic
                self.subscriber = rospy.Subscriber("/cv_camera/image_raw", Image, self.callback, queue_size=1)

                #self.detection(graph, category, score, expand)

                self.threshold = 0.6


    def callback(self, ros_data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        height, width, _ = cv_image.shape
        # Define Input and Ouput tensors

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        image = cv_image  # video_stream.read()
        image_expanded = np.expand_dims(image, axis=0)
        boxes, scores, classes, num = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            cv_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        frame_flag_detected = False
        center_flag_detected = False
        for i in range(0, num):
            score = scores[0][i]
            if score >= self.threshold:
                print(int(classes[0][i]))
                #if the whole frame is detected
                if int(classes[0][i]) == 1:
                    frame_flag_detected = True
                    ymin = boxes[0][i][0] * height
                    xmin = boxes[0][i][1] * width
                    ymax = boxes[0][i][2] * height
                    xmax = boxes[0][i][3] * width
                    centerX = (xmin+xmax)/2
                    centerY = (ymin+ymax)/2
                    #print("ymin: {}, ymax: {}, xmin: {}, xmax: {}".format(ymin, ymax, xmin, xmax))
                    #print("center x: {}\ncenter y: {}\nwidth: {}\nheight: {}".format(centerX, centerY, (xmax - xmin),
                    #                                                                 (ymax - ymin)))
                    self.find_relative_pose(centerX, centerY, (xmax - xmin), (xmax - xmin))
                #if center of frame is detected
                if int(classes[0][i]) == 2:
                    center_flag_detected = True
                    ymin = boxes[0][i][0] * height
                    xmin = boxes[0][i][1] * width
                    ymax = boxes[0][i][2] * height
                    xmax = boxes[0][i][3] * width
                    centerX = (xmin+xmax)/2
                    centerY = (ymin+ymax)/2
                    #print("ymin: {}, ymax: {}, xmin: {}, xmax: {}".format(ymin, ymax, xmin, xmax))
                    #print("center x: {}\ncenter y: {}\nwidth: {}\nheight: {}".format(centerX, centerY, (xmax - xmin),
                    #                                                                 (ymax - ymin)))
                    self.find_relative_pose_center(centerX, centerY, (xmax - xmin), (xmax - xmin))
        if not frame_flag_detected:
            self.cam_pose = Point()
            self.cam_pose.x = float("inf")
            self.cam_pose.y = float("inf")
            self.cam_pose.z = float("inf")
            self.cam_pose_pub.publish(self.cam_pose)
        if not center_flag_detected:
            self.cam_pose_center = Point()
            self.cam_pose_center.x = float("inf")
            self.cam_pose_center.y = float("inf")
            self.cam_pose_center.z = float("inf")
            self.cam_pose_center_pub.publish(self.cam_pose_center)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)
    def find_relative_pose(self, x, y, w, h):
        quad_3d = np.float32([[-0.345, -0.345, 0], [0.345, -0.345, 0], [0.345, 0.345, 0], [-0.345, 0.345, 0]])
        quad = np.float32(
            [[x - w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2], [x + w / 2, y - h / 2]])

        K = np.float64([[1472.512772, 0, 640.5],
                        [0, 1472.512772, 480.5],
                        [0.0, 0.0, 1.0]])


        dist_coef = np.zeros(4)
        _ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, K, dist_coef)
        rmat = cv2.Rodrigues(rvec)[0]
        #self.rotationMatrixToEulerAngles(rmat)
        cameraTranslatevector = -np.matrix(rmat).T * np.matrix(tvec)

        T0 = np.zeros((4, 4))
        T0[:3, :3] = rmat
        T0[:4, 3] = [0, 0, 0, 1]
        T0[:3, 3] = np.transpose(cameraTranslatevector)

        p0 = np.array([-0.345 / 2, -0.3011 / 2, 0, 1])
        z0 = np.dot(T0, p0)

        self.cam_pose.x = z0.item(0)
        self.cam_pose.y = z0.item(1)
        self.cam_pose.z = z0.item(2)
        self.cam_pose_pub.publish(self.cam_pose)
    def find_relative_pose_center(self, x, y, w, h):
        quad_3d = np.float32([[-0.11075, -0.11075, 0], [0.11075, -0.11075, 0], [0.11075, 0.11075, 0], [-0.11075, 0.11075, 0]])
        quad = np.float32(
            [[x - w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2], [x + w / 2, y - h / 2]])

        K = np.float64([[1472.512772, 0, 640.5],
                        [0, 1472.512772, 480.5],
                        [0.0, 0.0, 1.0]])


        dist_coef = np.zeros(4)
        _ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, K, dist_coef)
        rmat = cv2.Rodrigues(rvec)[0]
        #self.rotationMatrixToEulerAngles(rmat)
        cameraTranslatevector = -np.matrix(rmat).T * np.matrix(tvec)

        T0 = np.zeros((4, 4))
        T0[:3, :3] = rmat
        T0[:4, 3] = [0, 0, 0, 1]
        T0[:3, 3] = np.transpose(cameraTranslatevector)

        p0 = np.array([-0.2215 / 2, -0.2061 / 2, 0, 1])
        z0 = np.dot(T0, p0)

        self.cam_pose_center.x = z0.item(0)
        self.cam_pose_center.y = z0.item(1)
        self.cam_pose_center.z = z0.item(2)
        self.cam_pose_center_pub.publish(self.cam_pose_center)

    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):

        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        print("Rotation x: {}, y: {}, z: {}".format(x, y, z))
        return np.array([x, y, z])

    def _node_name(self,n):
      if n.startswith("^"):
        return n[1:]
      else:
        return n.split(":")[0]

    # Load a (frozen) Tensorflow model into memory.
    def load_frozenmodel(self):
        print('Loading frozen model into memory')
        if not split_model:
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return detection_graph, None, None
        else:
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded for ssd_mobilenet
            input_graph = tf.Graph()
            with tf.Session(graph=input_graph):
                if ssd_shape == 600:
                    shape = 7326
                else:
                    shape = 1917
                score = tf.placeholder(tf.float32, shape=(None, shape, num_classes),
                                       name="Postprocessor/convert_scores")
                expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
                for node in input_graph.as_graph_def().node:
                    if node.name == "Postprocessor/convert_scores":
                        score_def = node
                    if node.name == "Postprocessor/ExpandDims_1":
                        expand_def = node

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    dest_nodes = ['Postprocessor/convert_scores', 'Postprocessor/ExpandDims_1']

                    edges = {}
                    name_to_node_map = {}
                    node_seq = {}
                    seq = 0
                    for node in od_graph_def.node:
                        n = self._node_name(node.name)
                        name_to_node_map[n] = node
                        edges[n] = [self._node_name(x) for x in node.input]
                        node_seq[n] = seq
                        seq += 1

                    for d in dest_nodes:
                        assert d in name_to_node_map, "%s is not in graph" % d

                    nodes_to_keep = set()
                    next_to_visit = dest_nodes[:]
                    while next_to_visit:
                        n = next_to_visit[0]
                        del next_to_visit[0]
                        if n in nodes_to_keep:
                            continue
                        nodes_to_keep.add(n)
                        next_to_visit += edges[n]

                    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

                    nodes_to_remove = set()
                    for n in node_seq:
                        if n in nodes_to_keep_list: continue
                        nodes_to_remove.add(n)
                    nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

                    keep = graph_pb2.GraphDef()
                    for n in nodes_to_keep_list:
                        keep.node.extend([copy.deepcopy(name_to_node_map[n])])

                    remove = graph_pb2.GraphDef()
                    remove.node.extend([score_def])
                    remove.node.extend([expand_def])
                    for n in nodes_to_remove_list:
                        remove.node.extend([copy.deepcopy(name_to_node_map[n])])

                    with tf.device('/gpu:0'):
                        tf.import_graph_def(keep, name='')
                    with tf.device('/cpu:0'):
                        tf.import_graph_def(remove, name='')

            return detection_graph, score, expand

    def load_labelmap(self):
        print('Loading label map')
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index
def main():
    '''Initializes and cleanup ros node'''
    ic = tensorflow_detection()
    rospy.init_node('tensorflow_detection', anonymous=True)
    rospy.Rate(20)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image tensorflow detector x"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

#height, width, _ = cv_image.shape
#config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
#config.gpu_options.allow_growth = allow_memory_growth
#with self.graph.as_default():
#    with tf.Session(graph=self.graph, config=config) as sess:
#        # Define Input and Ouput tensors
#        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
#        detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
#        detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
#        detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
#         num_detections = self.graph.get_tensor_by_name('num_detections:0')
#
#         image = cv_image  # video_stream.read()
#         image_expanded = np.expand_dims(image, axis=0)
#         boxes, scores, classes, num = sess.run(
#             [detection_boxes, detection_scores, detection_classes, num_detections],
#             feed_dict={image_tensor: image_expanded})
#         vis_util.visualize_boxes_and_labels_on_image_array(
#             cv_image,
#             np.squeeze(boxes),
#             np.squeeze(classes).astype(np.int32),
#             np.squeeze(scores),
#             self.category_index,
#             use_normalized_coordinates=True,
#             line_thickness=8)
#         for i in range(0, num_detections):
#             score = scores[0][i]
#             if score >= self.threshold:
#                 ymin = boxes[0][i][0] * height
#                 xmin = boxes[0][i][1] * width
#                 ymax = boxes[0][i][2] * height
#                 xmax = boxes[0][i][3] * width
#                 centerX = xmin + ((xmax - xmin) / 2)
#                 centerY = ymin + ((ymax - ymin) / 2)
#                 print("center x: {}\ncenter y: {}\nwidth: {}\nheight: {}".format(centerX, centerY,
#                                                                                  (xmax - xmin),
#                                                                                  (ymax - ymin)))
#                 self.find_relative_pose(centerX, centerY, (xmax - xmin), (ymax - ymin))
#             else:
#                 self.cam_pose = Point()
#                 self.cam_pose.x = float("inf")
#                 self.cam_pose.y = float("inf")
#                 self.cam_pose.z = float("inf")
#                 self.cam_pose_pub.publish(self.cam_pose)
#         try:
#             self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
#         except CvBridgeError as e:
#             print(e)




