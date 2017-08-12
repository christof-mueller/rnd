#!/usr/bin/env python

# Import modules
from pcl_helper import *
from filter import *

def cluster_and_color(extracted_objects):
    white_cloud = XYZRGB_to_XYZ(extracted_objects)
    tree = white_cloud.make_kdtree()
	
	# Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(220)
    ec.set_MaxClusterSize(100000000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
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

def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # filtering
    cloud_filtered = filter_voxel_grid_downsampling(cloud, 0.01)
    cloud_filtered = filter_pass_through(cloud_filtered, 'z', 0.774, 1.2)
    extracted_objects, extracted_table = filter_RANSAC_PLANE(cloud_filtered, 0.001)
    
	# cluster points to objects and mark them with color
    cluster_objects = cluster_and_color(extracted_objects)

    # Convert PCL data to ROS messages
    pcl_objects_msg =  pcl_to_ros(extracted_objects)
    pcl_table_msg = pcl_to_ros(extracted_table)
    pcl_cluster_msg = pcl_to_ros(cluster_objects)

    # Publish ROS messages
    pcl_objects_pub.publish(pcl_objects_msg)
    pcl_table_pub.publish(pcl_table_msg)
    pcl_cluster_pub.publish(pcl_cluster_msg)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
    	rospy.spin()
