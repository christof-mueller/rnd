#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

from filter import *
from features import *

def colorize_objects(white_cloud, cluster_indices):
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
        	color_cluster_point_list.append([white_cloud[indice][0],
                                                 white_cloud[indice][1],
                                                 white_cloud[indice][2],
                                                 rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

def cluster(white_cloud):
    tree = white_cloud.make_kdtree()
	
	# Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(220)
    ec.set_MaxClusterSize(100000000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    
    return ec.Extract()

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # filtering
    cloud_filtered = filter_voxel_grid_downsampling(cloud, 0.01)
    cloud_filtered = filter_pass_through(cloud_filtered, 'z', 0.774, 1.2)
    extracted_objects_cloud, extracted_table_cloud = filter_RANSAC_PLANE(cloud_filtered, 0.001)
    
	# cluster points to objects and mark them with color
    white_cloud = XYZRGB_to_XYZ(extracted_objects_cloud)
    cluster_indices = cluster(white_cloud)
    colorized_objects = colorize_objects(white_cloud, cluster_indices)

    # Convert PCL data to ROS messages
    pcl_objects_msg =  pcl_to_ros(extracted_objects_cloud)
    pcl_table_msg = pcl_to_ros(extracted_table_cloud)
    pcl_cluster_msg = pcl_to_ros(colorized_objects)

    # Publish ROS messages
    pcl_objects_pub.publish(pcl_objects_msg)
    pcl_table_pub.publish(pcl_table_msg)
    pcl_cluster_pub.publish(pcl_cluster_msg)

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
    
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_objects_cloud.extract(pts_list)
        
        # convert the cluster from pcl to ROS using helper function
        object_to_classify = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(object_to_classify, using_hsv=True)
        nhists = compute_normal_histograms(get_normals(object_to_classify))
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = object_to_classify
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)


    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
