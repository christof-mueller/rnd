import pcl

def filter_outlier(cloud, mean_k, threshold):
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(mean_k)
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(threshold)
    return outlier_filter.filter()

def filter_voxel_grid_downsampling(cloud, leaf_size):
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return vox.filter()

def filter_pass_through(cloud, axis, limit_min, limit_max):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough.set_filter_limits(limit_min, limit_max)
    return passthrough.filter()

def filter_RANSAC_PLANE(cloud, distance_threshold):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.001)
    
    inliers, coefficients = seg.segment()
    out_points = cloud.extract(inliers, negative=True)
    in_points = cloud.extract(inliers, negative=False)
    return out_points, in_points
