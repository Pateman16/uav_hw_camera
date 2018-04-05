#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
based on @author: GustavZ real time object detection. Configured to be a ros node with custom trained network
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
import math

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream, SessionWorker

import time

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
width               = cfg['width']
height              = cfg['height']
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


# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(model_path):
        print('> Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'toy_frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('> Model found. Proceed.')

# helper function for split model
def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]

# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('> Loading frozen model into memory')
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
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
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
                dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
    
                edges = {}
                name_to_node_map = {}
                node_seq = {}
                seq = 0
                for node in od_graph_def.node:
                    n = _node_name(node.name)
                    name_to_node_map[n] = node
                    edges[n] = [_node_name(x) for x in node.input]
                    node_seq[n] = seq
                    seq += 1
                for d in dest_nodes:
                    assert d in name_to_node_map, "%s is not in graph" % d
    
                nodes_to_keep = set()
                next_to_visit = dest_nodes[:]
                
                while next_to_visit:
                    n = next_to_visit[0]
                    del next_to_visit[0]
                    if n in nodes_to_keep: continue
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


def load_labelmap():
    print('> Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detection(detection_graph, category_index, score, expand):
    print("> Building Graph")
    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    cur_frames = 0
    bridge = CvBridge()
    image_pub = rospy.Publisher("/tensorflow_detection/image", Image, queue_size=10)
    # topic where the coordinates go
    cam_pose_pub = rospy.Publisher("/tensorflow_detection/cam_point", Point, queue_size=1)
    cam_pose_center_pub = rospy.Publisher("/tensorflow_detection/cam_point_center", Point, queue_size=1)
    threshold = 0.85
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            if split_model:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                gpu_counter = 0
                cpu_counter = 0
            # Start Video Stream and FPS calculation
            fps = FPS2(fps_interval).start()
            video_stream = WebcamVideoStream(video_input,width,height).start()
            cur_frames = 0
            print("> Press 'q' to Exit")
            print('> Starting Detection')
            while video_stream.isActive():
                # actual Detection
                if split_model:
                    # split model in seperate gpu and cpu session threads
                    if gpu_worker.is_sess_empty():
                        # read video frame, expand dimensions and convert to rgb
                        image = video_stream.read()
                        image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                        # put new queue
                        gpu_feeds = {image_tensor: image_expanded}
                        if visualize:
                            gpu_extras = image # for visualization frame
                        else:
                            gpu_extras = None
                        gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)

                    g = gpu_worker.get_result_queue()
                    if g is None:
                        # gpu thread has no output queue. ok skip, let's check cpu thread.
                        gpu_counter += 1
                    else:
                        # gpu thread has output queue.
                        gpu_counter = 0
                        score,expand,image = g["results"][0],g["results"][1],g["extras"]

                        if cpu_worker.is_sess_empty():
                            # When cpu thread has no next queue, put new queue.
                            # else, drop gpu queue.
                            cpu_feeds = {score_in: score, expand_in: expand}
                            cpu_extras = image
                            cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)

                    c = cpu_worker.get_result_queue()
                    if c is None:
                        # cpu thread has no output queue. ok, nothing to do. continue
                        cpu_counter += 1
                        time.sleep(0.005)
                        continue # If CPU RESULT has not been set yet, no fps update
                    else:
                        cpu_counter = 0
                        boxes, scores, classes, num, image = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                else:
                    # default session
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})

                # Visualization of the results of a detection.
                if visualize:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    frame_flag_detected = False
                    center_flag_detected = False
                    for i in range(0, num):
                        score = scores[0][i]
                        if score >= threshold:
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
                                find_relative_pose(centerX, centerY, (xmax - xmin), (xmax - xmin), cam_pose_pub)
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
                                find_relative_pose_center(centerX, centerY, (xmax - xmin), (xmax - xmin), cam_pose_center_pub)
                    if not frame_flag_detected:
                        cam_pose = Point()
                        cam_pose.x = float("inf")
                        cam_pose.y = float("inf")
                        cam_pose.z = float("inf")
                        cam_pose_pub.publish(cam_pose)
                    if not center_flag_detected:
                        cam_pose_center = Point()
                        cam_pose_center.x = float("inf")
                        cam_pose_center.y = float("inf")
                        cam_pose_center.z = float("inf")
                        cam_pose_center_pub.publish(cam_pose_center)
                    if vis_text:
                        cv2.putText(image,"fps: {}".format(fps.fps_local()), (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    try:
                        image_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
                    except CvBridgeError as e:
                        print(e)
                    #cv2.imshow('object_detection', image)
                    # Exit Option
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                else:
                    cur_frames += 1
                    # Exit after max frames if no visualization
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if cur_frames%det_interval==0 and score > det_th:
                            label = category_index[_class]['name']
                            print("> label: {}\nscore: {}\nbox: {}".format(label, score, box))
                    if cur_frames >= max_frames:
                        break
                fps.update()

    # End everything
    if split_model:
        gpu_worker.stop()
        cpu_worker.stop()
    fps.stop()
    video_stream.stop()
    cv2.destroyAllWindows()
    print('> [INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('> [INFO] approx. FPS: {:.2f}'.format(fps.fps()))

def find_relative_pose(x, y, w, h, publisher):

    quad_3d = np.float32([[-0.345, -0.345, 0], [0.345, -0.345, 0], [0.345, 0.345, 0], [-0.345, 0.345, 0]])
    quad = np.float32([[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]])

    K = np.float64([[1150.2780757354, 0, 638.6271340537455],
                    [0, 1151.972786184758, 391.7481903792310],
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

    p0 = np.array([0.690/2, 0.610, 0, 1])
    z0 = np.dot(T0, p0)
    cam_pose = Point()
    cam_pose.x = -z0.item(0)
    cam_pose.y = z0.item(1)
    cam_pose.z = z0.item(2)
    publisher.publish(cam_pose)
def find_relative_pose_center(x, y, w, h, publisher):
    #quad_3d = np.float32([[0, 0, 0], [0.2215, 0, 0], [0.2215, 0.2215, 0], [0, 0.2215, 0]])
    #quad = np.float32(
    #    [[x - w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2], [x + w / 2, y - h / 2]])
    quad_3d = np.float32([[-0.11075, -0.11075, 0], [0.11075, -0.11075, 0], [0.11075, 0.11075, 0], [-0.11075, 0.11075, 0]])
    quad = np.float32(
        [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]])

    K = np.float64([[1150.2780757354, 0, 638.6271340537455],
                    [0, 1151.972786184758, 391.7481903792310],
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

    p0 = np.array([0.2950, 0.2215*2, 0, 1])
    z0 = np.dot(T0, p0)

    cam_pose = Point()
    cam_pose.x = -z0.item(0)
    cam_pose.y = z0.item(1)
    cam_pose.z = z0.item(2)
    publisher.publish(cam_pose)

def main():
    download_model()
    graph, score, expand = load_frozenmodel()
    category = load_labelmap()
    rospy.init_node('tensorflow_detection', anonymous=True)
    rospy.Rate(20)
    detection(graph, category, score, expand)


if __name__ == '__main__':
    main()
