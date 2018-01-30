#include "output_function.h"
#include "area_function.h"

//去除影像區域亮度
void DivideArea(InputArray _grayImage, InputArray _blurImage, OutputArray _divideImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat blurImage = _blurImage.getMat();
	CV_Assert(blurImage.type() == CV_8UC1);

	_divideImage.create(grayImage.size(), CV_8UC1);
	Mat divideImage = _divideImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			divideImage.at<uchar>(i, j) = (double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j)) * 255;
}

//大津閥值
void otsuThreshold(InputArray _grayImage, OutputArray _bwImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_bwImage.create(grayImage.size(), CV_8UC1);
	Mat bwImage = _bwImage.getMat();

	unsigned int *hist = new unsigned int[256]();
	for (size_t i = 0; i < grayImage.rows; ++i)
		for (size_t j = 0; j < grayImage.cols; ++j)
		{
			++hist[grayImage.at<uchar>(i, j)];
		}

	unsigned int sumB = 0;
	unsigned int sum1 = 0;
	float wB = 0.0f;
	float wF = 0.0f;
	float mF = 0.0f;
	float max_var = 0.0f;
	float inter_var = 0.0f;
	unsigned char th = 0;

	for (size_t i = 0; i < 256; ++i) {
		sum1 += i * hist[i];
	}

	for (size_t i = 0; i < 256; ++i)
	{
		wB += hist[i];
		wF = grayImage.rows*grayImage.cols - wB;
		if (wB == 0 || wF == 0) {
			continue;
		}
		sumB += i * hist[i];
		mF = (sum1 - sumB) / wF;
		inter_var = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
		if (inter_var >= max_var)
		{
			th = i;
			max_var = inter_var;
		}
	}

	delete[] hist;

	cv::threshold(grayImage, bwImage, th, 255, THRESH_BINARY);
}

//滯後閥值
void HysteresisThreshold(InputArray _grayImae, OutputArray _bwImage, int upperThreshold, int lowerThreshold)
{
	Mat grayImae = _grayImae.getMat();
	CV_Assert(grayImae.type() == CV_8UC1);

	_bwImage.create(grayImae.size(), CV_8UC1);
	Mat bwImage = _bwImage.getMat();

	Mat UT;		//上閥值二值化
	threshold(grayImae, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//下閥值二值化
	threshold(grayImae, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//弱邊緣
	MT.create(grayImae.size(), CV_8UC1);
	for (int i = 0; i < grayImae.rows; ++i)
		for (int j = 0; j < grayImae.cols; ++j)
		{
			if (LT.at<uchar>(i, j) == 255 && UT.at<uchar>(i, j) == 0)
				MT.at<uchar>(i, j) = 255;
			else
				MT.at<uchar>(i, j) = 0;

			if (UT.at<uchar>(i, j) == 255)
				bwImage.at<uchar>(i, j) = 255;
			else
				bwImage.at<uchar>(i, j) = 0;
		}

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 4);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum]();		// initialize label table with zero

	for (int i = 0; i < grayImae.rows; ++i)
		for (int j = 0; j < grayImae.cols; ++j)
		{
			//+ - + - + - +
			//| B | C | D |
			//+ - + - + - +
			//| E | A | F |
			//+ - + - + - +
			//| G | H | I |
			//+ - + - + - +

			int  C, E;

			if (i == 0) { C = 0; }
			else { C = LT.at<uchar>(i - 1, j); }

			if (j == 0) { E = 0; }
			else { E = LT.at<uchar>(i, j - 1); }

			// apply 4 connectedness  
			if (C || E)
			{
				++labeltable[labelImg.at<int>(i, j)];
			}
		}

	labeltable[0] = 0;		//clear 0 label

	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
		{
			if (labeltable[labelImg.at<int>(i, j)] > 0)
			{
				bwImage.at<uchar>(i, j) = 0;
			}
		}
	delete[] labeltable;
	labeltable = nullptr;
}