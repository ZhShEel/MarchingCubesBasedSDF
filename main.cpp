#include <iostream>
#include <fstream>
#include <cmath>
#include <pcl/surface/boost.h>
#include <pcl/surface/reconstruction.h>
#include <pcl/common/common.h>
#include <pcl/common/vector_average.h>
#include <pcl/Vertices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "fusionup.h"
/*#define sdf_voxel_count_  256*256*256
#define res_x_ 256
#define res_y_ 256
#define res_z_ 256*/
#define sdf_voxel_count_  128*128*128
#define res_x_ 128
#define res_y_ 128
#define res_z_ 128
#define iso_level_ 0
#define percentage_extend_grid_ 0

using namespace std;
const unsigned int edgeTable[256] = {
	0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};
const int triTable[256][16] = {
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 },
	{ 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 },
	{ 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 },
	{ 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 },
	{ 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 },
	{ 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 },
	{ 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 },
	{ 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 },
	{ 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 },
	{ 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1 },
	{ 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1 },
	{ 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1 },
	{ 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 },
	{ 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 },
	{ 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 },
	{ 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 },
	{ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 },
	{ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 },
	{ 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 },
	{ 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1 },
	{ 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1 },
	{ 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1 },
	{ 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1 },
	{ 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1 },
	{ 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1 },
	{ 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1 },
	{ 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1 },
	{ 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1 },
	{ 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1 },
	{ 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1 },
	{ 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1 },
	{ 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1 },
	{ 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1 },
	{ 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1 },
	{ 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1 },
	{ 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1 },
	{ 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1 },
	{ 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1 },
	{ 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
	{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 },
	{ 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 },
	{ 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 },
	{ 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 },
	{ 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 },
	{ 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 },
	{ 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 },
	{ 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 },
	{ 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 },
	{ 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 },
	{ 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 },
	{ 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 },
	{ 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 },
	{ 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 },
	{ 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 },
	{ 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1 },
	{ 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1 },
	{ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1 },
	{ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1 },
	{ 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1 },
	{ 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1 },
	{ 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1 },
	{ 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1 },
	{ 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1 },
	{ 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1 },
	{ 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1 },
	{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 },
	{ 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 },
	{ 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 },
	{ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 },
	{ 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1 },
	{ 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1 },
	{ 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1 },
	{ 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 },
	{ 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 },
	{ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 },
	{ 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1 },
	{ 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1 },
	{ 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1 },
	{ 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1 },
	{ 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1 },
	{ 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1 },
	{ 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1 },
	{ 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1 },
	{ 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1 },
	{ 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1 },
	{ 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1 },
	{ 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1 },
	{ 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1 },
	{ 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1 },
	{ 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1 },
	{ 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1 },
	{ 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1 },
	{ 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1 },
	{ 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1 },
	{ 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1 },
	{ 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1 },
	{ 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1 },
	{ 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1 },
	{ 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1 },
	{ 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1 },
	{ 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 },
	{ 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 },
	{ 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
	{ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};
ofstream o;
float* sdf_array_ = new float[sdf_voxel_count_]();

const int face_edge[6][4] = {
	{ 6,10,2,11 }, { 4, 5, 6, 7 }, { 0,9,4,8 },
	{ 0, 1, 2, 3 }, { 5,9,1,10 }, { 3,11,7,8 }
};

const int face_point[6][4] = {
	{ 7,6,2,3 }, { 4, 5, 6, 7 }, { 0, 1, 5,4 }, 
	{ 0, 1, 2, 3 }, { 6,5,1,2 }, { 0, 3, 7,4 }
};

float getGridValue(Eigen::Vector3i pos)
{
	/// TODO what to return?
	if (pos[0] < 0 || pos[0] >= res_x_)
		return -1;
	if (pos[1] < 0 || pos[1] >= res_y_)
		return -1;
	if (pos[2] < 0 || pos[2] >= res_z_)
		return -1;
	return sdf_array_[pos[0] * res_y_*res_z_ + pos[1] * res_z_ + pos[2]];
}

void LoadSdf(float* sdf_array, int sdf_voxel_count, string file_path) {
	//std::ifstream in(file_path, ios::in | ios::binary);
	std::ifstream in(file_path);
	if (!in.is_open()) {
		cerr << "Failed to load ";
		exit(1);
	}
	in.read((char *)sdf_array, sdf_voxel_count * sizeof(float));
}
void Loadtxt(float* sdf_array_, int volume, string file_path){
	std::ifstream in(file_path);
	for (int x = 0; x < volume; x++){
		in >> sdf_array_[x];
	}
}
void neigh1D(std::vector<float> &leaf, Eigen::Vector3i &index3d)
{
	leaf = std::vector<float>(8, 0);

	leaf[0] = getGridValue(index3d);
	leaf[1] = getGridValue(index3d + Eigen::Vector3i(1, 0, 0));
	leaf[2] = getGridValue(index3d + Eigen::Vector3i(1, 0, 1));
	leaf[3] = getGridValue(index3d + Eigen::Vector3i(0, 0, 1));
	leaf[4] = getGridValue(index3d + Eigen::Vector3i(0, 1, 0));
	leaf[5] = getGridValue(index3d + Eigen::Vector3i(1, 1, 0));
	leaf[6] = getGridValue(index3d + Eigen::Vector3i(1, 1, 1));
	leaf[7] = getGridValue(index3d + Eigen::Vector3i(0, 1, 1));

}
void interpolateEdge(Eigen::Vector3f &p1, Eigen::Vector3f &p2, float val_p1, float val_p2, Eigen::Vector3f &output)
{
	float val_p11 = (float)val_p1;
	float val_p22 = (float)val_p2;
	float mu = (0 - val_p11) / (val_p22 - val_p11);
	output = p1 + mu * (p2 - p1);
	//	cout << val_p11 << " " << val_p22<<" "<< mu << endl;
}
/*
void startdraw(float num1, float num2, float num3, float num4, Eigen::Vector3f &ax1, Eigen::Vector3f &ax2, Eigen::Vector3f &ax3, Eigen::Vector3f &ax4){
	Eigen::Vector3f ax12 = (ax1 + ax2)/2;
	Eigen::Vector3f ax23 = (ax3 + ax2) / 2;
	Eigen::Vector3f ax34 = (ax3 + ax4) / 2;
	Eigen::Vector3f ax14 = (ax1 + ax4) / 2;
	Eigen::Vector3f axmid = (ax1 + ax2+ax3+ax4) / 4;
	if (num1>0 && num3 > 0 && num2 < 0 && num4 < 0){
		startdraw(num1, (num1 + num2) / 2, (num1 + num2 + num3 + num4) / 4, (num1 + num4) / 2, ax1, ax12, axmid, ax14);
		startdraw((num1 + num2) / 2, num2, (num2 + num3) / 2, (num1 + num2 + num3 + num4) / 4, ax12,ax2, ax23,axmid);
		startdraw((num1 + num4) / 2, (num1 + num2 + num3 + num4) / 4, (num3 + num4) / 2,num4, ax14, axmid, ax34,ax4);
		startdraw((num1 + num2 + num3 + num4) / 4, (num2 + num3) / 2,num3,(num3+num4)/2, axmid, ax23,ax3,ax34);
	}
	else if (num1<0 && num3 < 0 && num2 > 0 && num4 > 0){
		startdraw(num1, (num1 + num2) / 2, (num1 + num2 + num3 + num4) / 4, (num1 + num4) / 2, ax1, ax12, axmid, ax14);
		startdraw((num1 + num2) / 2, num2, (num2 + num3) / 2, (num1 + num2 + num3 + num4) / 4, ax12, ax2, ax23, axmid);
		startdraw((num1 + num4) / 2, (num1 + num2 + num3 + num4) / 4, (num3 + num4) / 2, num4, ax14, axmid, ax34, ax4);
		startdraw((num1 + num2 + num3 + num4) / 4, (num2 + num3) / 2, num3, (num3 + num4) / 2, axmid, ax23, ax3, ax34);
	}
	else{
		if (num1*num2 < 0){

		}
	}
}*/

bool candraw(float num1, float num2, float num3, float num4, Eigen::Vector3f &ax1, Eigen::Vector3f &ax2, Eigen::Vector3f &ax3, Eigen::Vector3f &ax4){
	if (num1>0 && num3 > 0 && num2 < 0 && num4 < 0){
		return false;
	}
	else if (num1<0 && num3 < 0 && num2 > 0 && num4 > 0){
		return false;
	}
	else{
		return true;
	}
}
void drawline(std::vector<float> leaf, Eigen::Vector3f index3d, int len, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
		Eigen::Vector3f float_index[8];
		float_index[0] = index3d;
		float_index[1] = float_index[0] + Eigen::Vector3f(len, 0, 0);
		float_index[2] = float_index[0] + Eigen::Vector3f(len, 0, len);
		float_index[3] = float_index[0] + Eigen::Vector3f(0, 0, len);
		float_index[4] = float_index[0] + Eigen::Vector3f(0, len, 0);
		float_index[5] = float_index[0] + Eigen::Vector3f(len, len, 0);
		float_index[6] = float_index[0] + Eigen::Vector3f(len, len, len);
		float_index[7] = float_index[0] + Eigen::Vector3f(0, len, len);
		int flag = 0;
		for (int fa = 0; fa < 6; fa++){
			if (!candraw(leaf[face_point[fa][0]], leaf[face_point[fa][1]], leaf[face_point[fa][2]], leaf[face_point[fa][3]], float_index[face_point[fa][0]], float_index[face_point[fa][1]], float_index[face_point[fa][2]], float_index[face_point[fa][3]])){
				flag = 1;
				break;
			}
		}
		if (flag == 1){
			float small_leaf[19];
			Eigen::Vector3f small_3d;
			small_leaf[0] = (leaf[0] + leaf[3] + leaf[4] + leaf[7]) / 4;
			small_leaf[1] = (leaf[2] + leaf[3] + leaf[6] + leaf[7]) / 4;
			small_leaf[2] = (leaf[1] + leaf[2] + leaf[5] + leaf[6]) / 4;
			small_leaf[3] = (leaf[0] + leaf[1] + leaf[4] + leaf[5]) / 4;
			small_leaf[4] = (leaf[4] + leaf[5] + leaf[6] + leaf[7]) / 4;
			small_leaf[5] = (leaf[0] + leaf[1] + leaf[2] + leaf[3]) / 4;
			small_leaf[6] = (leaf[6] + leaf[7]) / 2;
			small_leaf[7] = (leaf[4] + leaf[7]) / 2;
			small_leaf[8] = (leaf[4] + leaf[5]) / 2;
			small_leaf[9] = (leaf[5] + leaf[6]) / 2;
			small_leaf[10] = (leaf[2] + leaf[6]) / 2;
			small_leaf[11] = (leaf[1] + leaf[5]) / 2;
			small_leaf[12] = (leaf[0] + leaf[4]) / 2;
			small_leaf[13] = (leaf[3] + leaf[7]) / 2;
			small_leaf[14] = (leaf[2] + leaf[3]) / 2;
			small_leaf[15] = (leaf[1] + leaf[2]) / 2;
			small_leaf[16] = (leaf[0] + leaf[1]) / 2;
			small_leaf[17] = (leaf[0] + leaf[3]) / 2;
			small_leaf[18] = (leaf[0] + leaf[1] + leaf[2] + leaf[3] + leaf[4] + leaf[5] + leaf[6] + leaf[7]) / 8;
			std::vector<float> di_leaf(8);
			di_leaf[0] = small_leaf[0];
			di_leaf[1] = small_leaf[18];
			di_leaf[2] = small_leaf[1];
			di_leaf[3] = small_leaf[13];
			di_leaf[4] = small_leaf[7];
			di_leaf[5] = small_leaf[4];
			di_leaf[6] = small_leaf[6];
			di_leaf[7] = leaf[7];
			small_3d = (float_index[0] + float_index[3] + float_index[4] + float_index[7]) / 4;
			drawline(di_leaf, small_3d, len / 2,cloud);
			di_leaf[0] = small_leaf[18];
			di_leaf[1] = small_leaf[2];
			di_leaf[2] = small_leaf[10];
			di_leaf[3] = small_leaf[1];
			di_leaf[4] = small_leaf[4];
			di_leaf[5] = small_leaf[9];
			di_leaf[6] = leaf[6];
			di_leaf[7] = small_leaf[6];
			small_3d = (float_index[0] + float_index[1] + float_index[2] + float_index[3] + float_index[4] + float_index[5] + float_index[6] + float_index[7]) / 8;
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = small_leaf[12];
			di_leaf[1] = small_leaf[3];
			di_leaf[2] = small_leaf[18];
			di_leaf[3] = small_leaf[0];
			di_leaf[4] = leaf[4];
			di_leaf[5] = small_leaf[8];
			di_leaf[6] = small_leaf[4];
			di_leaf[7] = small_leaf[7];
			small_3d = (float_index[0] + float_index[4]) / 2;
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = small_leaf[3];
			di_leaf[1] = small_leaf[11];
			di_leaf[2] = small_leaf[2];
			di_leaf[3] = small_leaf[18];
			di_leaf[4] = small_leaf[8];
			di_leaf[5] = leaf[5];
			di_leaf[6] = small_leaf[9];
			di_leaf[7] = small_leaf[4];
			small_3d = (float_index[0] + float_index[1] + float_index[4] + float_index[5]) / 4;
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = leaf[0];
			di_leaf[1] = small_leaf[16];
			di_leaf[2] = small_leaf[5];
			di_leaf[3] = small_leaf[17];
			di_leaf[4] = small_leaf[12];
			di_leaf[5] = small_leaf[3];
			di_leaf[6] = small_leaf[18];
			di_leaf[7] = small_leaf[0];
			small_3d = (float_index[0]);
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = small_leaf[16];
			di_leaf[1] = leaf[1];
			di_leaf[2] = small_leaf[15];
			di_leaf[3] = small_leaf[5];
			di_leaf[4] = small_leaf[3];
			di_leaf[5] = small_leaf[11];
			di_leaf[6] = small_leaf[2];
			di_leaf[7] = small_leaf[18];
			small_3d = (float_index[0] + float_index[1]) / 2;
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = small_leaf[17];
			di_leaf[1] = small_leaf[5];
			di_leaf[2] = small_leaf[14];
			di_leaf[3] = leaf[3];
			di_leaf[4] = small_leaf[0];
			di_leaf[5] = small_leaf[18];
			di_leaf[6] = small_leaf[1];
			di_leaf[7] = small_leaf[13];
			small_3d = (float_index[0] + float_index[3]) / 2;
			drawline(di_leaf, small_3d, len / 2, cloud);
			di_leaf[0] = small_leaf[5];
			di_leaf[1] = small_leaf[15];
			di_leaf[2] = leaf[2];
			di_leaf[3] = small_leaf[14];
			di_leaf[4] = small_leaf[18];
			di_leaf[5] = small_leaf[2];
			di_leaf[6] = small_leaf[10];
			di_leaf[7] = small_leaf[1];
			small_3d = (float_index[0] + float_index[1] + float_index[2] + float_index[3]) / 4;
			drawline(di_leaf, small_3d, len / 2, cloud);
		}
		else{
			int cubeindex = 0;
			Eigen::Vector3f vertex_list[12];
			if (leaf[0] < iso_level_) cubeindex |= 1;
			if (leaf[1] < iso_level_) cubeindex |= 2;
			if (leaf[2] < iso_level_) cubeindex |= 4;
			if (leaf[3] < iso_level_) cubeindex |= 8;
			if (leaf[4] < iso_level_) cubeindex |= 16;
			if (leaf[5] < iso_level_) cubeindex |= 32;
			if (leaf[6] < iso_level_) cubeindex |= 64;
			if (leaf[7] < iso_level_) cubeindex |= 128;
			// Cube is entirely in/out of the surface

			std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > p;
			p.resize(8);
			for (int i = 0; i < 8; ++i)
			{
				Eigen::Vector3f point = index3d;
				if (i & 0x4)
					point[1] = static_cast<float> (index3d[1] + 1);

				if (i & 0x2)
					point[2] = static_cast<float> (index3d[2] + 1);

				if ((i & 0x1) ^ ((i >> 1) & 0x1))
					point[0] = static_cast<float> (index3d[0] + 1);

				p[i] = point;
			}
			if (edgeTable[cubeindex] & 1)
				interpolateEdge(float_index[0], float_index[1], leaf[0], leaf[1], vertex_list[0]);
			if (edgeTable[cubeindex] & 2)
				interpolateEdge(float_index[1], float_index[2], leaf[1], leaf[2], vertex_list[1]);
			if (edgeTable[cubeindex] & 4)
				interpolateEdge(float_index[2], float_index[3], leaf[2], leaf[3], vertex_list[2]);
			if (edgeTable[cubeindex] & 8)
				interpolateEdge(float_index[3], float_index[0], leaf[3], leaf[0], vertex_list[3]);
			if (edgeTable[cubeindex] & 16)
				interpolateEdge(float_index[4], float_index[5], leaf[4], leaf[5], vertex_list[4]);
			if (edgeTable[cubeindex] & 32)
				interpolateEdge(float_index[5], float_index[6], leaf[5], leaf[6], vertex_list[5]);
			if (edgeTable[cubeindex] & 64)
				interpolateEdge(float_index[6], float_index[7], leaf[6], leaf[7], vertex_list[6]);
			if (edgeTable[cubeindex] & 128)
				interpolateEdge(float_index[7], float_index[4], leaf[7], leaf[4], vertex_list[7]);
			if (edgeTable[cubeindex] & 256)
				interpolateEdge(float_index[0], float_index[4], leaf[0], leaf[4], vertex_list[8]);
			if (edgeTable[cubeindex] & 512)
				interpolateEdge(float_index[1], float_index[5], leaf[1], leaf[5], vertex_list[9]);
			if (edgeTable[cubeindex] & 1024)
				interpolateEdge(float_index[2], float_index[6], leaf[2], leaf[6], vertex_list[10]);
			if (edgeTable[cubeindex] & 2048)
				interpolateEdge(float_index[3], float_index[7], leaf[3], leaf[7], vertex_list[11]);
			for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
			{
				pcl::PointXYZ p1, p2, p3;
				p1.x = vertex_list[triTable[cubeindex][i]][0];
				p1.y = vertex_list[triTable[cubeindex][i]][1];
				p1.z = vertex_list[triTable[cubeindex][i]][2];
				cloud.push_back(p1);
				p2.x = vertex_list[triTable[cubeindex][i + 1]][0];
				p2.y = vertex_list[triTable[cubeindex][i + 1]][1];
				p2.z = vertex_list[triTable[cubeindex][i + 1]][2];
				cloud.push_back(p2);
				p3.x = vertex_list[triTable[cubeindex][i + 2]][0];
				p3.y = vertex_list[triTable[cubeindex][i + 2]][1];
				p3.z = vertex_list[triTable[cubeindex][i + 2]][2];
				cloud.push_back(p3);
				o << "v " << p1.x << " " << p1.y << " " << p1.z << endl;
				o << "v " << p2.x << " " << p2.y << " " << p2.z << endl;
				o << "v " << p3.x << " " << p3.y << " " << p3.z << endl;
			}

		}
}
int main(){
	//LoadSdf(sdf_array_, sdf_voxel_count_, "mytesdf.txt");
	//Loadtxt(sdf_array_, sdf_voxel_count_, "mytesdf.txt");
	pcl::PointCloud<pcl::PointXYZ> cloud;

	pcl::PolygonMesh output;
	FusionUp fu;
	fu.Init();
	fu.LoadTUMTrajectory("groundtruth.txt");
	fu.readr_t();
	o.open("myanswer_50.obj");
	for (int x1 = 0; x1 < res_x_; x1++){
		for (int y1 = 0; y1 < res_y_; y1++){
			for (int z1 = 0; z1 < res_z_; z1++){
				sdf_array_[x1 * res_x_ * res_x_ + y1 *res_y_ + z1] = fu.TSDF_Volume[x1][y1][z1];
				//if (sdf_array_[x1 * res_x_ * res_x_ + y1 *res_y_ + z1]>0){
				//	cout <<x1<<" "<<y1<<" "<<z1<<" "<< sdf_array_[x1 * res_x_ * res_x_ + y1 *res_y_ + z1] << endl;
				//}
			}
		}
	}
	for (int x = 0; x < res_x_; ++x){
		for (int y = 0; y < res_y_; ++y){
			for (int z = 0; z < res_z_; ++z){
				Eigen::Vector3i index_3d(x, y, z);
				std::vector<float> leaf_node;
				neigh1D(leaf_node, index_3d);
				Eigen::Vector3f index_3df;
				index_3df[0] = (float)index_3d[0];
				index_3df[1] = (float)index_3d[1];
				index_3df[2] = (float)index_3d[2];
				bool havechanged = 1;
				if (fu.Weight_Volume[x][y][z] == 0){
					havechanged = 0;
				}
				if (x+1<res_x_&&fu.Weight_Volume[x+1][y][z] == 0){
					havechanged = 0;
				}
				if (x + 1<res_x_&&z + 1<res_z_&&fu.Weight_Volume[x + 1][y][z + 1] == 0){
					havechanged = 0;
				}
				if (z + 1<res_z_&&fu.Weight_Volume[x][y][z + 1] == 0){
					havechanged = 0;
				}
				if (y + 1<res_y_&&fu.Weight_Volume[x][y + 1][z] == 0){
					havechanged = 0;
				}
				if (x + 1<res_x_&&y + 1<res_y_&&fu.Weight_Volume[x + 1][y + 1][z] == 0){
					havechanged = 0;
				}
				if (x + 1<res_x_&&y+1<res_y_&&z + 1<res_z_&&fu.Weight_Volume[x + 1][y + 1][z + 1] == 0){
					havechanged = 0;
				}
				if (y+1<res_y_&&z + 1<res_z_&&fu.Weight_Volume[x][y + 1][z + 1] == 0){
					havechanged = 0;
				}
				if (havechanged == 1){
					//cout << "draw" << endl;
					drawline(leaf_node, index_3df, 1, cloud);
				}
				//if (sdf_array_[x * 128 * 128 + y * 128 + z] != -16384 && sdf_array_[x * 128 * 128 + y * 128 + z] != 0)
				// if (sdf_array_[x * 128 * 128 + y * 128 + z]<0 && sdf_array_[x * 128 * 128 + y * 128 + z]!=-16384)
			    //    cout << "Number:" << x<<" "<<y<<" "<<z<<":"<<sdf_array_[x * 128 * 128 + y * 128 + z] << endl;
				//		if (now[y][z].ax[0]!=0)
				 // 		cout << y<<" "<<z<<":"<<now[y][z].ax[0] << " " << now[y][z].ay[0] << " " << now[y][z].az[0] << endl;
				/////////////////////////////////////////////

			}
		}

	}
	for (int h = 0; h < cloud.size() / 3; ++h){
		o << "f " << 3 * h + 1 << " " << 3 * h + 2 << " " << 3 * h + 3 << endl;
	}
	o.close();


	//pcl::io::savePCDFileASCII("theanswer.pcd", cloud);
	//cout << sdf_array_[96876263] << endl;
	/*pcl::toPCLPointCloud2(cloud, output.cloud);

	output.polygons.resize(cloud.size() / 3);
	for (size_t i = 0; i < output.polygons.size(); ++i)
	{
	pcl::Vertices v;
	v.vertices.resize(3);
	for (int j = 0; j < 3; ++j)
	v.vertices[j] = static_cast<int> (i)* 3 + j;
	output.polygons[i] = v;
	}*/
	cout << "Done" << endl;
	system("pause");
	return 0;
}