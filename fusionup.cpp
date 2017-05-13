#include <iostream>
#include "fusionup.h"
#include <fstream>
//#include <unordered_map>
#include <opencv2/opencv.hpp>
#define VOLUMENUM 128
#define fx 525.0 
#define fy 525.0
#define cx 319.5
#define cy 239.5
#define trunsdis 0.04f
#define MAX_WEIGHT (1<<7)
#define sigma_space2_inv_half (1.0 / 72)
#define sigma_color2_inv_half (1.0/72)

using namespace std;
float InitR[] = {
	1, 0, 0,
	0, 1, 0,
	0, 0, 1
};
float InitT[] = {
	0, 0, 0
};
int it_=0;
void FusionUp::Init(){
	VolumeSize[0] = 0.02f*VOLUMENUM;
	VolumeSize[1] = 0.02f*VOLUMENUM;
	VolumeSize[2] = 0.02f*VOLUMENUM;
	CellSize[0] = 0.02f;
	CellSize[1] = 0.02f;
	CellSize[2] = 0.02f;
	for (int x = 0; x < VOLUMENUM; x++){
		for (int y = 0; y < VOLUMENUM; y++){
			for (int z = 0; z < VOLUMENUM; z++){
				TSDF_Volume[x][y][z] = 0;
				Weight_Volume[x][y][z] = 0;
			}
		}
	}
	mR = cv::Mat(3, 3, CV_32FC1);
	mt = cv::Mat(3, 1, CV_32FC1);
	mT = cv::Mat(4, 4, CV_32FC1);
	cout << "Init finished..." << endl;
}

void FusionUp::LoadTUMTrajectory(std::string dataset_path) {
	//std::ifstream list_stream(dataset_path + "depth_gt_associations.txt");
	std::ifstream list_stream(dataset_path);
	std::string  ts_gt;
	float  tx, ty, tz, qx, qy, qz, qw;
	double timestamp;
	while (list_stream >> timestamp
		>> tx >> ty >> tz
		>> qx >> qy >> qz >> qw) {
		cv::Mat wTc(4,4,CV_32FC1);
		//cout << tx << " " << ty << " " << tz << endl;
		wTc.at<float>(0, 0) = 1 - 2 * qy * qy - 2 * qz * qz;
		wTc.at<float>(0, 1) = 2 * qx * qy - 2 * qz * qw;
		wTc.at<float>(0, 2) = 2 * qx * qz + 2 * qy * qw;
		wTc.at<float>(0, 3) = tx;
		wTc.at<float>(1, 0) = 2 * qx * qy + 2 * qz * qw;
		wTc.at<float>(1, 1) = 1 - 2 * qx * qx - 2 * qz * qz;
		wTc.at<float>(1, 2) = 2 * qy * qz - 2 * qx * qw;
		wTc.at<float>(1, 3) = ty;
		wTc.at<float>(2, 0) = 2 * qx * qz - 2 * qy * qw;
		wTc.at<float>(2, 1) = 2 * qy * qz + 2 * qx * qw;
		wTc.at<float>(2, 2) = 1 - 2 * qx * qx - 2 * qy * qy;
		wTc.at<float>(2, 3) = tz;
		wTc.at<float>(3, 0) = 0;
		wTc.at<float>(3, 1) = 0;
		wTc.at<float>(3, 2) = 0;
		wTc.at<float>(3, 3) = timestamp - 1305031090;
		wTc_list.push_back(wTc);
	}
	cout << wTc_list.size() << " trans matrixs have been created." << endl;
	list_stream.close();
}
int xoosum = 0, xooin = 0;
void FusionUp::updateTSDF(cv::InputArray img_raw, cv::Mat T)
{
	cv::Mat T_inv = T.inv();///T.inv();
	ofstream aaaa("aaaa.txt");

	/*cout << "T:" << endl;
	for (int pp = 0; pp < T.rows; pp++){
		for (int ppp = 0; ppp < T.cols; ppp++){
			cout << T.at<float>(pp, ppp) << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "T.inv:" << endl;
	for (int pp = 0; pp < T.rows; pp++){
		for (int ppp = 0; ppp < T.cols; ppp++){
			cout << T_inv.at<float>(pp, ppp) << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	std::cout << T_inv << std::endl;
	//getchar();
	cv::Mat img = img_raw.getMat();
	for (int z = 0; z < VOLUMENUM; z++)
	{
		for (int x = 0; x < VOLUMENUM;x++)
		{
			for (int y = 0; y < VOLUMENUM; y++)
			{
				cv::Vec3f where(x, y, z);
				//cout << "cellsize:" << CellSize[0] << endl;
				where[0] *= CellSize[0];
				where[0] -= VolumeSize[0] / 2;

				where[1] *= CellSize[1];
				where[1] -= VolumeSize[1] / 2;

				where[2] *= CellSize[2];
				//where[2] -= VolumeSize[2] / 2;

				/*cv::Mat tr_where(3, 1, CV_32FC1);
				tr_where.at<float>(0, 0) = where[0];
				tr_where.at<float>(1, 0) = where[1];
				tr_where.at<float>(2, 0) = where[2];
				tr_where = R_inv*(tr_where - T);*/

				cv::Mat tr_where(4, 1, CV_32FC1);
				tr_where.at<float>(0, 0) = where[0];
				tr_where.at<float>(1, 0) = where[1];
				tr_where.at<float>(2, 0) = where[2];
				tr_where.at<float>(3, 0) = 1;
				//cout << "where:" << tr_where << endl;
				tr_where = T_inv*tr_where;
				//cout << "tr_where:" << tr_where << endl;

				//getchar();

				cv::Vec3f pic_coo_fux;
				pic_coo_fux[0] = tr_where.at<float>(0, 0);
				pic_coo_fux[1] = tr_where.at<float>(1, 0);
				pic_coo_fux[2] = tr_where.at<float>(2, 0);
				
				//std::cout << "pic_coo_fux: " << pic_coo_fux << std::endl;
				//getchar();

				cv::Vec2i x_coo;
				x_coo[0] = (int)((pic_coo_fux[1] * fy / pic_coo_fux[2]) + cy);//+cy wait for change
				x_coo[1] = (int)((pic_coo_fux[0] * fx / pic_coo_fux[2]) + cx);//+cx

				///std::cout << cx << " " << cy << " " << fx << " " << fy << std::endl;
				//cout << "x_coo:" << x_coo<< endl;
				//getchar();

				xoosum++;
				if (pic_coo_fux[2]>0&& x_coo[0] > 0 && x_coo[0]<img.rows && x_coo[1]>0 && x_coo[1] < img.cols)
				{
					xooin++;
					//scout <<"x_coo:"<< x_coo << endl;
					float Dp = img.at<unsigned short>(x_coo[0], x_coo[1]) / 5000.0f;
					//cout << Dp << endl;
					if (Dp != 0)
					{
						//float xl = (x_coo[1] - cx)*Dp / fx;
						//float yl = (x_coo[0] - cy)*Dp / fy;
						//float lambda_inv = 1.0 / sqrtf(xl*xl + yl*yl + 1);
						
						cv::Vec3f Tsub(T.at<float>(0, 3) - where[0], T.at<float>(1, 3) - where[1], T.at<float>(2, 3) - where[2]);
						//std::cout << "T0: " << T.col(3) << std::endl;
						//std::cout << "where: " << where << std::endl;
						//std::cout << sqrtf(Tsub[0] * Tsub[0] + Tsub[1] * Tsub[1] + Tsub[2] * Tsub[2]) << std::endl;
						//std::cout << "Dp: " << Dp << std::endl;

						//cv::Vec3f Tsub(tr_where.at<float>(0, 0), tr_where.at<float>(1, 0),tr_where.at<float>(2, 0));
						float sdf = sqrtf(Tsub[0] * Tsub[0] + Tsub[1] * Tsub[1] + Tsub[2] * Tsub[2]) - Dp;
						
						//cout << sqrtf(Tsub[0] * Tsub[0] + Tsub[1] * Tsub[1] + Tsub[2] * Tsub[2]) << "   " << Dp << endl;
						sdf = -sdf;

						//if (sdf < 0)
						//cout <<"sdf:"<< sdf << endl;
						if (sdf >= -trunsdis)
						{
							float tsdf = fmin(1.0f, sdf / trunsdis);
						//float tsdf = sdf / trunsdis;
							float weight_prev;
							float tsdf_prev;

							tsdf_prev = TSDF_Volume[x][y][z];
							weight_prev = Weight_Volume[x][y][z];
							Weight_Volume[x][y][z] = fmin(weight_prev + 1, MAX_WEIGHT);
							//cout << x << " " << y << " " << z << endl;
							TSDF_Volume[x][y][z] = (tsdf_prev*weight_prev + Weight_Volume[x][y][z] * tsdf) / (weight_prev + Weight_Volume[x][y][z]);
							//std::cout << "the former TSDF : " << tsdf_prev << "the weight_prev : " << weight_prev << std::endl;
						}
						/*float tsdf = -sdf / trunsdis;

						float weight_prev;
						float tsdf_prev;

						tsdf_prev = TSDF_Volume[x][y][z];
						weight_prev = Weight_Volume[x][y][z];
						Weight_Volume[x][y][z] = fmin(weight_prev + 1, MAX_WEIGHT);
						TSDF_Volume[x][y][z] = (tsdf_prev*weight_prev + Weight_Volume[x][y][z] * tsdf) / Weight_Volume[x][y][z];*/
						//if (TSDF_Volume[x][y][z]<0)
							//cout << "TSDF:" << TSDF_Volume[x][y][z] << endl;
						
					}
				}
				aaaa << TSDF_Volume[x][y][z] << " ";
				//std::cout << "the TSDF_VOLUME[" << x << "][" << y << "][" << z << "] = " << TSDF_Volume[x][y][z] << std::endl;
			}
		}
	}
	aaaa.close();
	//cout << "xoosum:" << xoosum << "  ,xooin:" << xooin << endl;
}

void FusionUp::readr_t()
{
	ifstream opendepthtxt("depth.txt");
	vector <float> times;
	vector<char*> which;
	for (int i = 0; i <50; i++)
	{
		cout << "reading pic:" << i + 1 << endl;
		//cv::Mat kankan = cvLoadImage("savedImage2.png", -1);
		double timestamp1;
		char whichdepth[50];
		opendepthtxt >> timestamp1 >> whichdepth;
		cv::Mat mDepth = cvLoadImage(whichdepth,-1);
		//std::cout << "type " << mDepth.type() << std::endl;
		
		float timestamp2 = timestamp1 - 1305031090;
		//cout << whichdepth << endl;
		
		while (wTc_list[it_].at<float>(3, 3) < timestamp2){
			it_++;
		}
		wTc_list[it_].at<float>(3, 3) = 1;
		wTc_list[0].at<float>(3, 3) = 1;
		mT = wTc_list[0].inv()*wTc_list[it_];

		//cout << "it_" << it_ << endl;
		for (int x = 0; x < mT.rows; x++){
			for (int y = 0; y < mT.cols; y++){
				cout << mT.at<float>(x, y) << " ";
			}
			cout << endl;
		}
		/*for (int x = 0; x < mDepth.rows; x++){
			for (int y = 0; y < mDepth.cols; y++){
				if (mDepth.at<double>(x, y) != 0){
					cout << x << " " << y << " ";
					cout << mDepth.at<double>(x, y) << endl;
				}
			}
		}*/

		/*char openpos[19] = { 'P', 'o', 's' };
		openpos[3] = opennumchar1[0];
		openpos[4] = opennumchar2[0];
		openpos[5] = '.';
		openpos[6] = 't';
		openpos[7] = 'x';
		openpos[8] = 't';
		ifstream ff;
		ff.open(openpos);
		double tex[16];
		for (int inin = 0; inin < 16; inin++){
			ff >> tex[inin];
		}
		ff.close();*/

		/*mR.at<float>(0, 0) = tex[0];  mR.at<float>(0, 1) = tex[4];   mR.at<float>(0, 2) = tex[8];
		mR.at<float>(1, 0) = tex[1];  mR.at<float>(1, 1) = tex[5];   mR.at<float>(1, 2) = tex[9];
		mR.at<float>(2, 0) = tex[2];  mR.at<float>(2, 1) = tex[6];   mR.at<float>(2, 2) = tex[10];
		mt.at<float>(0, 0) = tex[12];  mt.at<float>(1, 0) = tex[13];   mt.at<float>(2, 0) = tex[14];*/
		/*mT.at<float>(0, 0) = tex[0];  mT.at<float>(0, 1) = tex[4];   mT.at<float>(0, 2) = tex[8];  mT.at<float>(0, 3) = tex[12];
		mT.at<float>(1, 0) = tex[1];  mT.at<float>(1, 1) = tex[5];   mT.at<float>(1, 2) = tex[9];  mT.at<float>(1, 3) = tex[13];
		mT.at<float>(2, 0) = tex[2];  mT.at<float>(2, 1) = tex[6];   mT.at<float>(2, 2) = tex[10];  mT.at<float>(2, 3) = tex[14];
		mT.at<float>(3, 0) = tex[3];  mT.at<float>(3, 1) = tex[7];   mT.at<float>(3, 2) = tex[11];  mT.at<float>(3, 3) = tex[15];*/
		//mR = mR.inv();
		cout << "updating TSDF..." << endl;
		
		updateTSDF(mDepth, mT);
	}
	opendepthtxt.close();
}


