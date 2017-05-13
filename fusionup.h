#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
#define VOLUMENUM 128
#define fx 525.0 
#define fy 525.0
#define cx 319.5
#define cy 239.5
#define MAX_WEIGHT (1<<7)

class FusionUp{
public:
	FusionUp(){
		TSDF_Volume = new float**[VOLUMENUM];
		Weight_Volume = new float**[VOLUMENUM];
		for (int i = 0; i < VOLUMENUM; i++)
		{
			TSDF_Volume[i] = new float*[VOLUMENUM];
			Weight_Volume[i] = new float*[VOLUMENUM];
			for (int j = 0; j < VOLUMENUM; j++)
			{
				TSDF_Volume[i][j] = new float[VOLUMENUM];
				memset(TSDF_Volume[i][j], 0, VOLUMENUM*sizeof(float));
				Weight_Volume[i][j] = new float[VOLUMENUM];
				memset(Weight_Volume[i][j], 0, VOLUMENUM*sizeof(float));
			}
		}
	}
	float*** TSDF_Volume;
	float*** Weight_Volume;
	cv::Vec3f VolumeSize, CellSize;

	void Init();
	void pred(cv::InputArray src, cv::OutputArray dst);
	void updateTSDF(cv::InputArray TSDF, cv::Mat T);
	void readr_t();
	void FusionUp::LoadTUMTrajectory(std::string dataset_path);
	cv::Mat mR;
	cv::Mat mt,mT;
	cv::Mat mDepth;
	std::vector<cv::Mat> wTc_list;
};