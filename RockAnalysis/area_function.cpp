#include "output_function.h"
#include "area_function.h"

/*基於面的分割混合模式*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat mixImage = _mixImage.getMat();
	CV_Assert(mixImage.type() == CV_8UC1);

	_divideImage.create(grayImage.size(), CV_8UC1);
	Mat divideImage = _divideImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			divideImage.at<uchar>(i, j) = (double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j)) * 255;
}

/*滯後閥值*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold, int lowerThreshold)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	_bwLine.create(gradm.size(), CV_8UC1);
	Mat bwLine = _bwLine.getMat();

	Mat UT;		//上閥值二值化
	threshold(gradm, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//下閥值二值化
	threshold(gradm, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//弱邊緣
	MT.create(gradm.size(), CV_8UC1);
	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			if (LT.at<uchar>(i, j) == 255 && UT.at<uchar>(i, j) == 0)
				MT.at<uchar>(i, j) = 255;
			else
				MT.at<uchar>(i, j) = 0;

			if (UT.at<uchar>(i, j) == 255)
				bwLine.at<uchar>(i, j) = 255;
			else
				bwLine.at<uchar>(i, j) = 0;
		}

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 8);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum];		// initialize label table with zero  
	memset(labeltable, 0, labelNum * sizeof(int));

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
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

			if (i == 0 || j == gradm.cols - 1) { D = 0; }
			else { D = UT.at<uchar>(i - 1, j + 1); }

			if (j == 0) { E = 0; }
			else { E = UT.at<uchar>(i, j - 1); }

			if (j == gradm.cols - 1) { F = 0; }
			else { F = UT.at<uchar>(i, j + 1); }

			if (i == gradm.rows - 1 || j == 0) { G = 0; }
			else { G = UT.at<uchar>(i + 1, j - 1); }

			if (i == gradm.rows - 1) { H = 0; }
			else { H = UT.at<uchar>(i + 1, j); }

			if (i == gradm.rows - 1 || j == gradm.cols - 1) { I = 0; }
			else { I = UT.at<uchar>(i + 1, j + 1); }

			// apply 8 connectedness  
			if (B || C || D || E || F || G || H || I)
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
				bwLine.at<uchar>(i, j) = 255;
			}
		}
	delete[] labeltable;
	labeltable = nullptr;
}

/*去除雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears, bool BW)
{
	Mat binaryImg = _binaryImg.getMat();
	CV_Assert(binaryImg.type() == CV_8UC1);

	_clearAreaImage.create(binaryImg.size(), CV_8UC1);
	Mat clearAreaImage = _clearAreaImage.getMat();

	Mat labels(binaryImg.size(), CV_32SC1, Scalar(0));

	// 0 claer black noise
	// 1 clear wite noise
	if (BW == 0)
	{
		for (int i = 0; i < binaryImg.rows; i++)
			for (int j = 0; j < binaryImg.cols; j++)
				if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
				else { binaryImg.at<uchar>(i, j) = 255; }
	}

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  

	int *labeltable = new int[binaryImg.rows*binaryImg.cols];		// initialize label table with zero  
	memset(labeltable, 0, binaryImg.rows*binaryImg.cols * sizeof(int));
	int ntable = 0;

	//	labeling scheme
	//	+ - + - + - +
	//	| D | C | E |
	//	+ - + - + - +
	//	| B | A |   |
	//	+ - + - + - +
	//	A is the center pixel of a neighborhood.In the 3 versions of connectedness :
	//	4 : A connects to B and C
	//	6 : A connects to B, C, and D
	//	8 : A connects to B, C, D, and E


	for (int i = 0; i < binaryImg.rows; i++)
		for (int j = 0; j < binaryImg.cols; j++)
			if (binaryImg.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == binaryImg.cols - 1) { E = 0; }
				else { E = findroot(labeltable, labels.at<int>(i - 1, j + 1)); }

				if (nears == 4)		// apply 4 connectedness  
				{
					if (B && C)	// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							labeltable[C] = B;
							labels.at<int>(i, j) = B;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }            // B is object but C is not  
					else if (C) { labels.at<int>(i, j) = C; }            // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; }	// B, C, D not object - new object label and put into table  
				}
				else if (nears == 6)	// apply 6 connected ness  
				{
					if (D) { labels.at<int>(i, j) = D; }              // D object, copy label and move on  
					else if (B && C)		// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							int tlabel = B < C ? B : C;
							labeltable[B] = tlabel;
							labeltable[C] = tlabel;
							labels.at<int>(i, j) = tlabel;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }        // B is object but C is not  	
					else if (C) { labels.at<int>(i, j) = C; }        // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } 	// B, C, D not object - new object label and put into table
				}
				else if (nears == 8)	// apply 8 connectedness  
				{
					if (B || C || D || E)
					{
						int tlabel;
						if (B) { tlabel = B; }
						else if (C) { tlabel = C; }
						else if (D) { tlabel = D; }
						else if (E) { tlabel = E; }

						labels.at<int>(i, j) = tlabel;

						if (B && B != tlabel) { labeltable[B] = tlabel; }
						if (C && C != tlabel) { labeltable[C] = tlabel; }
						if (D && D != tlabel) { labeltable[D] = tlabel; }
						if (E && E != tlabel) { labeltable[E] = tlabel; }
					}
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } // label and put into table
				}
			}
			else { labels.at<int>(i, j) = 0; }	// A is not an object so leave it

			for (int i = 0; i <= ntable; i++)
				labeltable[i] = findroot(labeltable, i);	// consolidate component table  

			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

																				// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > noise) { labeltable[i] = 255; }	// number the objects from 1 through n objects  and reset label table
				else { labeltable[i] = 0; }

				// run through the look-up table again  
				for (int i = 0; i < binaryImg.rows; i++)
					for (int j = 0; j < binaryImg.cols; j++)
						clearAreaImage.at<uchar>(i, j) = labeltable[labels.at<int>(i, j)];

				delete[] labeltable;
				labeltable = nullptr;

				if (BW == 0)
				{
					for (int i = 0; i < clearAreaImage.rows; i++)
						for (int j = 0; j < clearAreaImage.cols; j++)
						{
							if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
							else { binaryImg.at<uchar>(i, j) = 255; }

							if (clearAreaImage.at<uchar>(i, j) == 255) { clearAreaImage.at<uchar>(i, j) = 0; }
							else { clearAreaImage.at<uchar>(i, j) = 255; }
						}
				}
}