#include "output_function.h"
#include "line_function.h"

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady) {

	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_gradx.create(grayImage.size(), CV_16SC1);
	Mat gradx = _gradx.getMat();

	_grady.create(grayImage.size(), CV_16SC1);
	Mat grady = _grady.getMat();

	Mat grayImageRef;
	copyMakeBorder(grayImage, grayImageRef, 1, 1, 1, 1, BORDER_REPLICATE);
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j) {
			gradx.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i + 1, j) + (float)grayImageRef.at<uchar>(i + 1, j + 2))*0.5;
			grady.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i, j + 1) + (float)grayImageRef.at<uchar>(i + 2, j + 1))*0.5;
		}
}

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _gradx, InputArray _grady, OutputArray _gradf) {

	Mat gradx = _gradx.getMat();
	CV_Assert(gradx.type() == CV_16SC1);

	Mat grady = _grady.getMat();
	CV_Assert(grady.type() == CV_16SC1);

	_gradf.create(grady.rows, gradx.cols, CV_16SC2);
	Mat gradf = _gradf.getMat();

	for (int i = 0; i < grady.rows; ++i)
		for (int j = 0; j < gradx.cols; ++j)
		{
			gradf.at<Vec2s>(i, j)[0] = gradx.at<short>(i, j);
			gradf.at<Vec2s>(i, j)[1] = grady.at<short>(i, j);
		}

}

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradf, OutputArray _gradm, OutputArray _gradd)
{
	Mat gradf = _gradf.getMat();
	CV_Assert(gradf.type() == CV_16SC2);

	_gradm.create(gradf.size(), CV_8UC1);
	Mat gradm = _gradm.getMat();

	_gradd.create(gradf.size(), CV_32FC1);
	Mat gradd = _gradd.getMat();

	for (int i = 0; i < gradf.rows; ++i)
		for (int j = 0; j < gradf.cols; ++j)
		{
			short x = gradf.at<Vec2s>(i, j)[0];
			short y = gradf.at<Vec2s>(i, j)[1];
			gradm.at<uchar>(i, j) = sqrt(x*x + y*y);

			if (x == 0 && y == 0) { gradd.at<float>(i, j) = -1000.0f; }	//用以顯示無梯度方向
			else { gradd.at<float>(i, j) = atan2(y, x); }
		}
}

/*基於線的分割混合模式*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradmblur = _gradmblur.getMat();
	CV_Assert(gradmblur.type() == CV_8UC1);

	_gradmDivide.create(gradm.size(), CV_8UC1);
	Mat gradmDivide = _gradmDivide.getMat();

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
			gradmDivide.at<uchar>(i, j) = ((double)gradmblur.at<uchar>(i, j) / (double)gradm.at<uchar>(i, j) >= 1 || gradm.at<uchar>(i, j) == 0 || gradmblur.at<uchar>(i, j) == 0) ? 0 : (1 - (double)gradmblur.at<uchar>(i, j) / (double)gradm.at<uchar>(i, j)) * 255;
}

/*滯後切割*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC)
{
	Mat lineHT = _lineHT.getMat();
	CV_Assert(lineHT.type() == CV_8UC1);

	Mat area = _area.getMat();
	CV_Assert(area.type() == CV_8UC1);

	_lineHC.create(lineHT.size(), CV_8UC1);
	Mat lineHC = _lineHC.getMat();

	Mat UT(lineHT.size(), CV_8UC1, Scalar(0));		//上閥值

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (area.at<uchar>(i, j) == 0 && lineHT.at<uchar>(i, j) == 255)
				UT.at<uchar>(i, j) = 255;

	Mat MT(lineHT.size(), CV_8UC1, Scalar(0));	//中閥值
	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (area.at<uchar>(i, j) == 255 && lineHT.at<uchar>(i, j) == 255)
				MT.at<uchar>(i, j) = 255;

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 4);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum];		// initialize label table with zero  
	memset(labeltable, 0, labelNum * sizeof(int));

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
		{
			//+ - + - + - +
			//| B | C | D |
			//+ - + - + - +
			//| E | A | F |
			//+ - + - + - +
			//| G | H | I |
			//+ - + - + - +

			int B, C, D, E, F, G, H, I;

			if (i == 0 || j == 0) { B = 0; }
			else { B = UT.at<uchar>(i - 1, j - 1); }

			if (i == 0) { C = 0; }
			else { C = UT.at<uchar>(i - 1, j); }

			if (i == 0 || j == lineHT.cols - 1) { D = 0; }
			else { D = UT.at<uchar>(i - 1, j + 1); }

			if (j == 0) { E = 0; }
			else { E = UT.at<uchar>(i, j - 1); }

			if (j == lineHT.cols - 1) { F = 0; }
			else { F = UT.at<uchar>(i, j + 1); }

			if (i == lineHT.rows - 1 || j == 0) { G = 0; }
			else { G = UT.at<uchar>(i + 1, j - 1); }

			if (i == lineHT.rows - 1) { H = 0; }
			else { H = UT.at<uchar>(i + 1, j); }

			if (i == lineHT.rows - 1 || j == lineHT.cols - 1) { I = 0; }
			else { I = UT.at<uchar>(i + 1, j + 1); }

			// apply 8 connectedness  
			if (B || C || D || E || F || G || H || I)
			{
				++labeltable[labelImg.at<int>(i, j)];
			}
		}

	labeltable[0] = 0;		//clear 0 label

	Mat mask(lineHT.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
			if (labeltable[labelImg.at<int>(i, j)] > 0 || UT.at<uchar>(i, j) == 255) { mask.at<uchar>(i, j) = 255; }
	delete[] labeltable;
	labeltable = nullptr;

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (mask.at<uchar>(i, j) == 255) { lineHC.at<uchar>(i, j) = 255; }
			else { lineHC.at<uchar>(i, j) = 0; }
}
