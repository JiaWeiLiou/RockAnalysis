#include "output_function.h"

//尋找根結點
int findroot(int labeltable[], int label)
{
	int x = label;
	while (x != labeltable[x])
		x = labeltable[x];
	return x;
}

//尋找連通物
int bwlabel(InputArray _bwImage, OutputArray _labels, int nears)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_labels.create(bwImage.size(), CV_32SC1);
	Mat labels = _labels.getMat();
	labels = Scalar(0);

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  
	int *labeltable = new int[bwImage.rows*bwImage.cols]();		// initialize label table with zero  
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


	for (int i = 0; i < bwImage.rows; i++)
		for (int j = 0; j < bwImage.cols; j++)
			if (bwImage.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == bwImage.cols - 1) { E = 0; }
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

			for (int i = 0; i < bwImage.rows; i++)
				for (int j = 0; j < bwImage.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

			// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < bwImage.rows; i++)
				for (int j = 0; j < bwImage.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > 0)
					labeltable[i] = ++nobj;	// number the objects from 1 through n objects  and reset label table

			// run through the look-up table again  
			for (int i = 0; i < bwImage.rows; i++)
				for (int j = 0; j < bwImage.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];

			delete[] labeltable;
			labeltable = nullptr;
			return nobj;
}

//創建色環
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//紅色(Red)     至黃色(Yellow)
	int YG = 15;	//黃色(Yellow)  至綠色(Green)
	int GC = 15;	//綠色(Green)   至青色(Cyan)
	int CB = 15;	//青澀(Cyan)    至藍色(Blue)
	int BM = 15;	//藍色(Blue)    至洋紅(Magenta)
	int MR = 15;	//洋紅(Magenta) 至紅色(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

//創建色條
void makecolorbar(vector<Scalar> &colorbar)
{
	vector<Scalar> maincolor;

	maincolor.push_back(Scalar(0, 0, 127.5));      //深藍色
	maincolor.push_back(Scalar(0, 0, 255));		   //藍色
	maincolor.push_back(Scalar(0, 127.5, 255));	   //青色至藍色
	maincolor.push_back(Scalar(0, 255, 255));	   //青色
	maincolor.push_back(Scalar(0, 255, 127.5));	   //綠色至青色
	maincolor.push_back(Scalar(0, 255, 0));		   //綠色
	maincolor.push_back(Scalar(127.5, 255, 0));	   //黃色至綠色
	maincolor.push_back(Scalar(255, 255, 0));	   //黃色
	maincolor.push_back(Scalar(255, 127.5, 0));	   //紅色至黃色
	maincolor.push_back(Scalar(255, 0, 0));		   //紅色
	maincolor.push_back(Scalar(127.5, 0, 0));      //深紅色

	int layer = 15;		//各漸層漸變階層數

	for (int i = 0; i < maincolor.size() - 1; i++)
	{
		for (int j = 0; j < layer; j++)
		{
			double r = maincolor[i][0] + (maincolor[i + 1][0] - maincolor[i][0]) / layer*j;
			double g = maincolor[i][1] + (maincolor[i + 1][1] - maincolor[i][1]) / layer*j;
			double b = maincolor[i][2] + (maincolor[i + 1][2] - maincolor[i][2]) / layer*j;
			colorbar.push_back(Scalar(r, g, b));
		}
	}
}

//將灰階圖像轉以灰階色條顯示並自動拉伸
void DrawGrayBar(InputArray _grayImage, OutputArray _graybarImage)
{
	Mat grayImage;
	Mat temp = _grayImage.getMat();

	if (temp.type() == CV_16SC1) { convertScaleAbs(temp, temp); }
	if (temp.type() == CV_8UC1) { temp.convertTo(grayImage, CV_32FC1); }
	else { grayImage = _grayImage.getMat(); }	//CV_32FC1

	_graybarImage.create(grayImage.size(), CV_8UC1);
	Mat graybarImage = _graybarImage.getMat();

	// determine range
	float minvalue = 0;
	float maxvalue = 0;

	// find max  and min value
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
		{
			minvalue = minvalue < grayImage.at<float>(i, j) ? minvalue : grayImage.at<float>(i, j);
			maxvalue = maxvalue > grayImage.at<float>(i, j) ? maxvalue : grayImage.at<float>(i, j);
		}

	//linear stretch from 0 to 255
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			graybarImage.at<uchar>(i, j) = (grayImage.at<float>(i, j) - minvalue) / (maxvalue - minvalue) * 255;
}

//將灰階圖像轉以紅藍色條顯示並自動拉伸
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage)
{
	Mat grayImage;
	Mat temp = _grayImage.getMat();
	if (temp.type() == CV_8UC1) { temp.convertTo(grayImage, CV_32FC1); }
	else if (temp.type() == CV_16SC1) { convertScaleAbs(temp, grayImage); }
	else { grayImage = _grayImage.getMat(); }	//CV_32FC1

	_colorbarImage.create(grayImage.size(), CV_8UC3);
	Mat colorbarImage = _colorbarImage.getMat();

	static vector<Scalar> colorbar; //Scalar i,g,b  
	if (colorbar.empty()) { makecolorbar(colorbar); }

	// determine range
	float minvalue = 0;
	float maxvalue = 0;

	// find max  and min value
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
		{
			minvalue = minvalue < grayImage.at<float>(i, j) ? minvalue : grayImage.at<float>(i, j);
			maxvalue = maxvalue > grayImage.at<float>(i, j) ? maxvalue : grayImage.at<float>(i, j);
		}

	//linear stretch from 0 to 255
	for (int i = 0; i < colorbarImage.rows; ++i)
		for (int j = 0; j < colorbarImage.cols; ++j)
		{
			uchar *data = colorbarImage.data + colorbarImage.step[0] * i + colorbarImage.step[1] * j;

			float fk = (grayImage.at<float>(i, j) - minvalue) / (maxvalue - minvalue) * (colorbar.size() - 1);  //計算灰度值對應之索引位置
			int k0 = floor(fk);
			int k1 = ceil(fk);
			float f = fk - k0;

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorbar[k0][b] / 255.0f;
				float col1 = colorbar[k1][b] / 255.0f;
				float col = (1 - f) * col0 + f * col1;
				data[2 - b] = 255.0f * col;
			}
		}
}

//將梯度圖像轉以色環方向顯示
void DrawColorRing(InputArray _gradf, OutputArray _colorringImage)
{
	Mat gradf = _gradf.getMat();

	_colorringImage.create(gradf.size(), CV_8UC3);
	Mat colorringImage = _colorringImage.getMat();

	static vector<Scalar> colorwheel; //Scalar i,g,b  
	if (colorwheel.empty()) { makecolorwheel(colorwheel); }

	if (gradf.type() == CV_16SC2)	//梯度場域
	{
		// determine motion range: 
		int maxrad = 0;
		int minrad = 0;

		// Find max flow to normalize fx and fy  
		for (int i = 0; i < gradf.rows; ++i)
			for (int j = 0; j < gradf.cols; ++j)
			{
				Vec2s gradf_at_point = gradf.at<Vec2s>(i, j);
				float fx = gradf_at_point[0];
				float fy = gradf_at_point[1];
				float rad = sqrt(fx * fx + fy * fy);
				maxrad = maxrad > rad ? maxrad : rad;
				minrad = minrad < rad ? minrad : rad;
			}

		for (int i = 0; i < gradf.rows; ++i)
			for (int j = 0; j < gradf.cols; ++j)
			{
				uchar *data = colorringImage.data + colorringImage.step[0] * i + colorringImage.step[1] * j;
				Vec2s gradf_at_point = gradf.at<Vec2s>(i, j);

				float fx = gradf_at_point[0];
				float fy = gradf_at_point[1];

				if (fx == 0 && fy == 0) { for (int b = 0; b < 3; ++b) { data[b] = 255; } }
				else
				{
					float rad = (sqrt(fx * fx + fy * fy) - minrad) / (maxrad - minrad);

					float angle = atan2(fy, fx) / CV_PI;    //單位為-1至+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
					int k0 = (int)fk;
					int k1 = (k0 + 1) % colorwheel.size();
					float f = fk - k0;

					for (int b = 0; b < 3; b++)
					{
						float col0 = colorwheel[k0][b];
						float col1 = colorwheel[k1][b];
						float col = 255 - rad*(255 - ((1 - f) * col0 + f * col1));	// increase saturation with radius 
						data[2 - b] = (int)col;
					}
				}
			}
	}
	else //gradf.type() == CV_32FC1 (梯度方向)
	{
		for (int i = 0; i < gradf.rows; ++i)
			for (int j = 0; j < gradf.cols; ++j)
			{
				uchar *data = colorringImage.data + colorringImage.step[0] * i + colorringImage.step[1] * j;

				//用以顯示無梯度方向
				if (gradf.at<float>(i, j) == -1000.0f) { for (int b = 0; b < 3; ++b) { data[b] = 255; } }
				else
				{
					float angle = gradf.at<float>(i, j) / CV_PI;    //單位為-1至+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
					int k0 = (int)fk;
					int k1 = (k0 + 1) % colorwheel.size();
					float f = fk - k0;

					for (int b = 0; b < 3; b++)
					{
						float col0 = colorwheel[k0][b];
						float col1 = colorwheel[k1][b];
						float col = (1 - f) * col0 + f * col1;
						data[2 - b] = (int)col;
					}
				}
			}
	}
}

//將二值圖像轉以標籤顯示
void DrawLabel(InputArray _bwImage, OutputArray _labelImage)
{
	Mat bwImage;
	Mat temp = _bwImage.getMat();
	CV_Assert(temp.type() == CV_8UC1 || temp.type() == CV_32FC1);
	if (temp.type() == CV_32FC1) { temp.convertTo(bwImage, CV_8UC1); }
	else { bwImage = temp; }

	_labelImage.create(bwImage.size(), CV_8UC3);
	Mat labelImage = _labelImage.getMat();

	Mat object(bwImage.size(), CV_8UC1);
	for (int i = 0; i < bwImage.rows; i++)
		for (int j = 0; j < bwImage.cols; j++)
			if (bwImage.at<uchar>(i, j) == 255) { object.at<uchar>(i, j) = 255; }
			else { object.at<uchar>(i, j) = 0; }

			Mat labels;
			int objectNum = bwlabel(object, labels, 4);

			RNG rng(12345);
			vector<Scalar> color;

			color.push_back(Scalar(0, 0, 0));
			for (int i = 1; i <= objectNum; i++) { color.push_back(Scalar(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255))); }	//防止深色看不清楚

			for (int i = 0; i < bwImage.rows; i++)
				for (int j = 0; j < bwImage.cols; j++)
				{
					labelImage.at<Vec3b>(i, j)[0] = color[labels.at<int>(i, j)][0];
					labelImage.at<Vec3b>(i, j)[1] = color[labels.at<int>(i, j)][1];
					labelImage.at<Vec3b>(i, j)[2] = color[labels.at<int>(i, j)][2];
				}
}

//將二值圖像結合原始影像轉以疊合圖像顯示
void DrawImage(InputArray _bwImage, InputArray _image, OutputArray _combineImage)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	Mat image = _image.getMat();

	if (image.type() == CV_8UC1)
	{
		_combineImage.create(image.size(), CV_8UC3);
		Mat combineImage = _combineImage.getMat();

		for (int i = 0; i < bwImage.rows; i++)
			for (int j = 0; j < bwImage.cols; j++)
			{
				if (bwImage.at<uchar>(i, j) == 0)
				{
					combineImage.at<Vec3b>(i, j)[0] = 255;
					combineImage.at<Vec3b>(i, j)[1] = 0;
					combineImage.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					combineImage.at<Vec3b>(i, j)[0] = image.at<uchar>(i, j);
					combineImage.at<Vec3b>(i, j)[1] = image.at<uchar>(i, j);
					combineImage.at<Vec3b>(i, j)[2] = image.at<uchar>(i, j);
				}
			}
	}
	else if (image.type() == CV_8UC3)
	{
		_combineImage.create(image.size(), CV_8UC3);
		Mat combineImage = _combineImage.getMat();

		for (int i = 0; i < bwImage.rows; i++)
			for (int j = 0; j < bwImage.cols; j++)
			{
				if (bwImage.at<uchar>(i, j) == 0)
				{
					combineImage.at<Vec3b>(i, j)[0] = 255;
					combineImage.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1];
					combineImage.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2];
				}
				else
				{
					combineImage.at<Vec3b>(i, j)[0] = image.at<Vec3b>(i, j)[0];
					combineImage.at<Vec3b>(i, j)[1] = image.at<Vec3b>(i, j)[1];
					combineImage.at<Vec3b>(i, j)[2] = image.at<Vec3b>(i, j)[2];
				}
			}
	}
}

//將種子點結合原物件轉以疊合圖像顯示
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineImae)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectSeed = _objectSeed.getMat();
	CV_Assert(objectSeed.type() == CV_8UC1);

	_combineImae.create(object.size(), CV_8UC1);
	Mat combineImae = _combineImae.getMat();

	for (int i = 0; i < combineImae.rows; ++i)
		for (int j = 0; j < combineImae.cols; ++j)
		{
			if (object.at<uchar>(i, j) == 255) { combineImae.at<uchar>(i, j) = 128; }
			if (objectSeed.at<uchar>(i, j) == 255) { combineImae.at<uchar>(i, j) = 255; }
			if (object.at<uchar>(i, j) == 0) { combineImae.at<uchar>(i, j) = 0; }
		}
}

//將二值圖像擬合橢圓並顯示
vector<Size2f> DrawEllipse(InputArray _object, OutputArray _ellipseImage)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	_ellipseImage.create(object.size(), CV_8UC1);
	Mat ellipseImage = _ellipseImage.getMat();

	for (int i = 0; i < ellipseImage.rows; ++i)
		for (int j = 0; j < ellipseImage.cols; ++j)
			ellipseImage.at<uchar>(i, j) = 0;

	Mat labels;
	int objectNum = bwlabel(object, labels, 4);

	Mat elementO = (Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	Mat objectErode;
	morphologyEx(object, objectErode, MORPH_ERODE, elementO);

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			labels.at<int>(i, j) = objectErode.at<uchar>(i,j) ? 0 : labels.at<int>(i, j);

	vector<vector<Point2i>> pointset;
	for (int i = 0; i < objectNum - 1; ++i) { pointset.push_back(vector<Point2i>()); }

	for (int i = 0; i < labels.rows; ++i)
		for (int j = 0; j < labels.cols; ++j)
			if (labels.at<int>(i, j) != 0)
				pointset[labels.at<int>(i, j) - 1].push_back(Point2i(j, i));

	vector<Size2f> ellipse_param;

	for (int i = 0; i < objectNum - 1; ++i)
		if (pointset[i].size() > 5)
		{
			RotatedRect ellipse_obj = fitEllipse(pointset[i]);

			//畫出擬合的橢圓
			ellipse(ellipseImage, ellipse_obj, Scalar(255), 1, CV_AA);

			//儲存橢圓參數
			ellipse_param.push_back(ellipse_obj.size);
		}

	return ellipse_param;
}