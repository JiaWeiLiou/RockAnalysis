#include "output_function.h"
#include "object_function.h"

PixelElement::PixelElement(float i, int j, int k)
{
	value = i;
	x = j;
	y = k;
}

/*影像形態學重建*/
void Reconstruct(InputArray _marker, InputArray _mask, OutputArray _objectR)
{
	Mat marker = _marker.getMat();
	CV_Assert(marker.type() == CV_32FC1);

	Mat mask = _mask.getMat();
	CV_Assert(mask.type() == CV_32FC1);

	_objectR.create(mask.size(), CV_32FC1);
	Mat objectR = _objectR.getMat();

	min(marker, mask, objectR);
	dilate(objectR, objectR, Mat());
	min(objectR, mask, objectR);
	Mat temp1 = Mat(marker.size(), CV_8UC1);
	Mat temp2 = Mat(marker.size(), CV_8UC1);
	do
	{
		objectR.copyTo(temp1);
		dilate(objectR, objectR, Mat());
		min(objectR, mask, objectR);
		compare(temp1, objectR, temp2, CV_CMP_NE);
	} while (sum(temp2).val[0] != 0);
}

/*H-min 轉換*/
void HMinimaTransform(InputArray _objectDT, OutputArray _objectHMT, float H)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectHMT.create(objectDT.size(), CV_32FC1);
	Mat objectHMT = _objectHMT.getMat();

	Mat mask = objectDT;

	Mat marker(objectDT.size(), CV_32FC1);
	for (int i = 0; i < mask.rows; ++i)
		for (int j = 0; j < mask.cols; ++j)
			marker.at<float>(i, j) = mask.at<float>(i, j) - H;

	Reconstruct(marker, mask, objectHMT);
}

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _objectLM)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectLM.create(objectDT.size(), CV_8UC1);
	Mat objectLM = _objectLM.getMat();

	for (int i = 0; i < objectLM.rows; ++i)
		for (int j = 0; j < objectLM.cols; ++j)
			objectLM.at<uchar>(i, j) = 255;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectLM.rows; ++y)
		for (int x = 0; x < objectLM.cols; ++x)
			if (objectLM.at<uchar>(y, x) == 255)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectDT.cols) && (y + dy >= 0) && (y + dy < objectDT.rows))
						{
							if (floor(objectDT.at<float>(y, x)) > floor(objectDT.at<float>(y + dy, x + dx)))  // If pe2.value < pe1.value, pe1 is not a local minimum
							{
								objectLM.at<uchar>(y, x) = 0;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectDT.cols) && (yh + dyh >= 0) && (yh + dyh < objectDT.rows))
												if (objectLM.at<uchar>(yh + dyh, xh + dxh) == 255)
												{
													if (floor(objectDT.at<float>(yh + dyh, xh + dxh)) == floor(objectDT.at<float>(y, x)))
													{
														objectLM.at<uchar>(yh + dyh, xh + dxh) = 0;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
}

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue, float precision)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_label.create(objectDT.size(), CV_32SC1);
	Mat label = _label.getMat();

	for (int i = 0; i < label.rows; ++i)
		for (int j = 0; j < label.cols; ++j)
			label.at<int>(i, j) = LABEL_UNPROCESSED;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectDT.rows; ++y)
		for (int x = 0; x < objectDT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectDT.cols) && (y + dy >= 0) && (y + dy < objectDT.rows))
						{
							if (floor(objectDT.at<float>(y, x) / precision) > floor(objectDT.at<float>(y + dy, x + dx) / precision))  // If pe2.value < pe1.value, pe1 is not a local minimum
							{
								label.at<int>(y, x) = LABEL_NOLOCALMINIMUM;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectDT.cols) && (yh + dyh >= 0) && (yh + dyh < objectDT.rows))
												if (label.at<int>(yh + dyh, xh + dxh) == LABEL_UNPROCESSED)
												{
													if (floor(objectDT.at<float>(yh + dyh, xh + dxh) / precision) == floor(objectDT.at<float>(y, x) / precision))
													{
														label.at<int>(yh + dyh, xh + dxh) = LABEL_NOLOCALMINIMUM;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
	delete mpFifo;

	mpFifo = new queue<Point2i>();
	int mNumberOfLabels = LABEL_MINIMUM;

	for (int y = 0; y < objectDT.rows; ++y)
		for (int x = 0; x < objectDT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				label.at<int>(y, x) = mNumberOfLabels;
				mpFifo->push(Point2i(x, y));

				while (!(mpFifo->empty()))
				{
					Point2i fifoElement = mpFifo->front();
					int xf = fifoElement.x;
					int yf = fifoElement.y;

					for (int dyf = -1; dyf <= 1; ++dyf)
						for (int dxf = -1; dxf <= 1; ++dxf)
							if ((xf + dxf >= 0) && (xf + dxf < objectDT.cols) && (yf + dyf >= 0) && (yf + dyf < objectDT.rows))
							{
								if (label.at<int>(yf + dyf, xf + dxf) == LABEL_UNPROCESSED)
								{
									label.at<int>(yf + dyf, xf + dxf) = mNumberOfLabels;
									mpFifo->push(Point2i(xf + dxf, yf + dyf));
								}
								else if (label.at<int>(yf + dyf, xf + dxf) == LABEL_NOLOCALMINIMUM)
								{
									label.at<int>(yf + dyf, xf + dxf) = LABEL_PROCESSING;
									mvSortedQueue.push(PixelElement(objectDT.at<float>(yf + dyf, xf + dxf), xf + dxf, yf + dyf));
								}
							}

					mpFifo->pop();
				}
				++mNumberOfLabels;
			}
	delete mpFifo;
}

/*最小值*/
void MinimaDetection(InputArray _objectIM, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue)
{
	Mat objectIM = _objectIM.getMat();
	CV_Assert(objectIM.type() == CV_32FC1);

	_label.create(objectIM.size(), CV_32SC1);
	Mat label = _label.getMat();

	float min = 0.0f;
	for (int i = 0; i < objectIM.rows; ++i)
		for (int j = 0; j < objectIM.cols; ++j)
			if (objectIM.at<float>(i, j) < min) { min = objectIM.at<float>(i, j); }

	for (int i = 0; i < label.rows; ++i)
		for (int j = 0; j < label.cols; ++j)
			label.at<int>(i, j) = LABEL_UNPROCESSED;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectIM.rows; ++y)
		for (int x = 0; x < objectIM.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectIM.cols) && (y + dy >= 0) && (y + dy < objectIM.rows))
						{
							if (objectIM.at<float>(y, x) != min)
							{
								label.at<int>(y, x) = LABEL_NOLOCALMINIMUM;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectIM.cols) && (yh + dyh >= 0) && (yh + dyh < objectIM.rows))
												if (label.at<int>(yh + dyh, xh + dxh) == LABEL_UNPROCESSED)
												{
													if (objectIM.at<float>(yh + dyh, xh + dxh) == objectIM.at<float>(y, x))
													{
														label.at<int>(yh + dyh, xh + dxh) = LABEL_NOLOCALMINIMUM;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
	delete mpFifo;

	mpFifo = new queue<Point2i>();
	int mNumberOfLabels = LABEL_MINIMUM;

	for (int y = 0; y < objectIM.rows; ++y)
		for (int x = 0; x < objectIM.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				label.at<int>(y, x) = mNumberOfLabels;
				mpFifo->push(Point2i(x, y));

				while (!(mpFifo->empty()))
				{
					Point2i fifoElement = mpFifo->front();
					int xf = fifoElement.x;
					int yf = fifoElement.y;

					for (int dyf = -1; dyf <= 1; ++dyf)
						for (int dxf = -1; dxf <= 1; ++dxf)
							if ((xf + dxf >= 0) && (xf + dxf < objectIM.cols) && (yf + dyf >= 0) && (yf + dyf < objectIM.rows))
							{
								if (label.at<int>(yf + dyf, xf + dxf) == LABEL_UNPROCESSED)
								{
									label.at<int>(yf + dyf, xf + dxf) = mNumberOfLabels;
									mpFifo->push(Point2i(xf + dxf, yf + dyf));
								}
								else if (label.at<int>(yf + dyf, xf + dxf) == LABEL_NOLOCALMINIMUM)
								{
									label.at<int>(yf + dyf, xf + dxf) = LABEL_PROCESSING;
									mvSortedQueue.push(PixelElement(objectIM.at<float>(yf + dyf, xf + dxf), xf + dxf, yf + dyf));
								}
							}

					mpFifo->pop();
				}
				++mNumberOfLabels;
			}
	delete mpFifo;
}

/*檢查相鄰區域是否存在標籤*/
bool CheckForAlreadyLabeledNeighbours(int x, int y, Mat &label, Point2i &outLabeledNeighbour, int &outLabel)
{
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
				if (label.at<int>(y + dy, x + dx) > LABEL_RIDGE)
				{
					outLabeledNeighbour = Point2i(x + dx, y + dy);
					outLabel = label.at<int>(y + dy, x + dx);
					return true;
				}
	return false;
}

/*檢查是否為分水嶺*/
bool CheckIfPixelIsWatershed(int x, int y, Mat &label, Point2i &inLabeledNeighbour, int &inLabelOfNeighbour)
{
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
				if ((label.at<int>(y + dy, x + dx) >= LABEL_MINIMUM) && (label.at<int>(y + dy, x + dx) != label.at<int>(inLabeledNeighbour.y, inLabeledNeighbour.x)) && ((inLabeledNeighbour.x != x + dx) || (inLabeledNeighbour.y != y + dy)))
				{
					label.at<int>(y, x) = LABEL_RIDGE;
					return true;
				}
	return false;
}

/*擴展區域最小值*/
void ExtendLocalMinimaDetection(InputArray _objectDT, OutputArray _objectEM, float H)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectEM.create(objectDT.size(), CV_8UC1);
	Mat objectEM = _objectEM.getMat();

	Mat objectHMT;
	HMinimaTransform(objectDT, objectHMT, H);

	for (int i = 0; i < objectHMT.rows; ++i)
		for (int j = 0; j < objectHMT.cols; ++j)
			objectHMT.at<float>(i, j) = -objectHMT.at<float>(i, j);

	LocalMinimaDetection(objectHMT, objectEM);
}

/*距離閥值*/
void DistanceCut(InputArray _objectDT, OutputArray _objectDC)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectDC.create(objectDT.size(), CV_8UC1);
	Mat objectDC = _objectDC.getMat();

	Mat objectHMT;
	HMinimaTransform(objectDT, objectHMT, 3);

	int imageMinLength = objectDT.rows < objectDT.cols ? objectDT.rows : objectDT.cols;

	// threshold
	for (int i = 0; i < objectHMT.rows; ++i)
		for (int j = 0; j < objectHMT.cols; ++j)
			objectDC.at<uchar>(i, j) = objectHMT.at<float>(i, j) > imageMinLength * 0.025 ? 255 : 0;
}

/*增加未標記之標籤*/
void AddLabel(InputArray _object, InputArray _objectSeed, OutputArray _objectAL)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectSeed = _objectSeed.getMat();
	CV_Assert(objectSeed.type() == CV_8UC1);

	_objectAL.create(object.size(), CV_8UC1);
	Mat objectAL = _objectAL.getMat();

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (object.at<uchar>(i, j) == 0) { objectSeed.at<uchar>(i, j) = 0; }
	
	objectSeed.copyTo(objectAL);

	Mat labels;
	int num = bwlabel(object, labels, 4) + 1;	//include label 0

	int *labeltable = new int[num]();  

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (objectSeed.at<uchar>(i, j) != 0) { labeltable[labels.at<int>(i, j)] = 1; }

	labeltable[0] = 1;

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (labeltable[labels.at<int>(i, j)] == 0) 
			{ 
				objectAL.at<uchar>(i, j) = 255; 
			}
	delete[] labeltable;
	labeltable = nullptr;
}

/*加深低窪區*/
void ImposeMinima(InputArray _objectDT, InputArray _objectLC, OutputArray _objectIM)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	Mat objectLC = _objectLC.getMat();
	CV_Assert(objectLC.type() == CV_8UC1);

	_objectIM.create(objectDT.size(), CV_32FC1);
	Mat objectIM = _objectIM.getMat();

	float min = 0;
	for (int i = 0; i < objectDT.rows; ++i)
		for (int j = 0; j < objectDT.cols; ++j)
		{
			objectDT.at<float>(i, j) = -objectDT.at<float>(i, j);
			if (objectDT.at<float>(i, j) < min) { min = objectDT.at<float>(i, j); }
		}

	min = min - 10;
	objectDT.copyTo(objectIM);
			
	for (int i = 0; i < objectDT.rows; ++i)
		for (int j = 0; j < objectDT.cols; ++j)
			if (objectLC.at<uchar>(i, j) != 0) { objectIM.at<float>(i, j) = min; }
}

/*分水嶺轉換*/
void WatershedTransform(InputArray _object, InputArray _objectIM, OutputArray _objectWT)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectIM = _objectIM.getMat();
	CV_Assert(objectIM.type() == CV_32FC1);

	_objectWT.create(objectIM.size(), CV_8UC1);
	Mat objectWT = _objectWT.getMat();

	Mat label;
	priority_queue<PixelElement, vector<PixelElement>, mycomparison> mvSortedQueue;
	MinimaDetection(objectIM, label, mvSortedQueue);

	while (!mvSortedQueue.empty())
	{
		PixelElement lItemA = mvSortedQueue.top();
		mvSortedQueue.pop();
		int labelOfNeighbour = 0;

		// (a) Pop element and find positive labeled neighbour
		Point2i alreadyLabeledNeighbour;
		int x = lItemA.x;
		int y = lItemA.y;

		// (b) Check if current pixel is watershed pixel by checking if there are different finally labeled neighbours
		if (CheckForAlreadyLabeledNeighbours(x, y, label, alreadyLabeledNeighbour, labelOfNeighbour))
			if (!(CheckIfPixelIsWatershed(x, y, label, alreadyLabeledNeighbour, labelOfNeighbour)))
			{
				// c) if not watershed pixel, assign label of neighbour and add the LABEL_NOLOCALMINIMUM neighbours to priority_queue for processing
				/*UpdateLabel(x, y, objectIM, label, labelOfNeighbour, mvSortedQueue);*/

				label.at<int>(y, x) = labelOfNeighbour;

				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
							if (label.at<int>(y + dy, x + dx) == LABEL_NOLOCALMINIMUM)
							{
								label.at<int>(y + dy, x + dx) = LABEL_PROCESSING;
								mvSortedQueue.push(PixelElement(objectIM.at<float>(y + dy, x + dx), x + dx, y + dy));
							}
			}
	}

	object.copyTo(objectWT);

	// d) finalize the labelImage
	for (int y = 0; y < objectWT.rows; ++y)
		for (int x = 0; x < objectWT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_RIDGE) { objectWT.at<uchar>(y, x) = 0; }
}

/*結合線與面的二值邊緣*/
void BWCombine(InputArray _area, InputArray _line, OutputArray _object)
{
	Mat area = _area.getMat();
	CV_Assert(area.type() == CV_8UC1);

	Mat line = _line.getMat();
	CV_Assert(line.type() == CV_8UC1);

	_object.create(area.size(), CV_8UC1);
	Mat object = _object.getMat();

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (area.at<uchar>(i, j) == 0 || line.at<uchar>(i, j) == 0) { object.at<uchar>(i, j) = 0; }
			else { object.at<uchar>(i, j) = 255; }
}

/*反轉二值圖*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_bwImageR.create(bwImage.size(), CV_8UC1);
	Mat bwImageR = _bwImageR.getMat();

	for (int i = 0; i < bwImage.rows; ++i)
		for (int j = 0; j < bwImage.cols; ++j)
			if (bwImage.at<uchar>(i, j) == 0) { bwImageR.at<uchar>(i, j) = 255; }
			else { bwImage.at<uchar>(i, j) = 0; }
}

/*填補空洞*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_bwFillhole.create(bwImage.size(), CV_8UC1);
	Mat bwFillhole = _bwFillhole.getMat();

	Mat holeImg(bwImage.rows + 2, bwImage.cols + 2, CV_8UC1, 255);
	for (int i = 1; i < holeImg.rows - 1; ++i)
		for (int j = 1; j < holeImg.cols - 1; ++j)
			holeImg.at<uchar>(i, j) = bwImage.at<uchar>(i - 1, j - 1) ? 0 : 255;

	Mat labelImg;
	int num = bwlabel(holeImg, labelImg, 8);

	int *labeltable = new int[num + 1]();

	// calculate obj size
	for (size_t i = 0; i < labelImg.rows; ++i)
		for (size_t j = 0; j < labelImg.cols; ++j)
			++labeltable[labelImg.at<int>(i, j)];

	// find max size
	int maxSize = 0;
	for (size_t i = 1; i < num + 1; ++i)
		maxSize = labeltable[i] > maxSize ? labeltable[i] : maxSize;

	int tolSize = ceil((double)maxSize * 0.001);

	for (size_t i = 0; i < labelImg.rows; ++i)
		for (size_t j = 0; j < labelImg.cols; ++j)
			if (labeltable[labelImg.at<int>(i, j)] < tolSize)
				holeImg.at<uchar>(i, j) = 0;

	for (int i = 0; i < bwFillhole.rows; ++i)
		for (int j = 0; j < bwFillhole.cols; ++j)
			bwFillhole.at<uchar>(i, j) = holeImg.at<uchar>(i + 1, j + 1) ? 0 : 255;

	delete[] labeltable;
}

//去除雜訊
void ClearNoise(InputArray _bwImage, OutputArray _clearAreaImage, int noise, int nears, bool BW)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_clearAreaImage.create(bwImage.size(), CV_8UC1);
	Mat clearAreaImage = _clearAreaImage.getMat();

	Mat labels(bwImage.size(), CV_32SC1, Scalar(0));

	// 0 claer black noise
	// 1 clear wite noise
	if (BW == 0)
	{
		for (int i = 0; i < bwImage.rows; i++)
			for (int j = 0; j < bwImage.cols; j++)
				if (bwImage.at<uchar>(i, j) == 255) { bwImage.at<uchar>(i, j) = 0; }
				else { bwImage.at<uchar>(i, j) = 255; }
	}

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
				if (labeltable[i] > noise) { labeltable[i] = 255; }	// number the objects from 1 through n objects  and reset label table
				else { labeltable[i] = 0; }

				// run through the look-up table again  
				for (int i = 0; i < bwImage.rows; i++)
					for (int j = 0; j < bwImage.cols; j++)
						clearAreaImage.at<uchar>(i, j) = labeltable[labels.at<int>(i, j)];

				delete[] labeltable;
				labeltable = nullptr;

				if (BW == 0)
				{
					for (int i = 0; i < clearAreaImage.rows; i++)
						for (int j = 0; j < clearAreaImage.cols; j++)
						{
							if (bwImage.at<uchar>(i, j) == 255) { bwImage.at<uchar>(i, j) = 0; }
							else { bwImage.at<uchar>(i, j) = 255; }

							if (clearAreaImage.at<uchar>(i, j) == 255) { clearAreaImage.at<uchar>(i, j) = 0; }
							else { clearAreaImage.at<uchar>(i, j) = 255; }
						}
				}
}

//疊加灰階影像
void AddGray(InputArray _objectDT, InputArray _gray, OutputArray _objectAG)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	Mat gray = _gray.getMat();
	CV_Assert(gray.type() == CV_8UC1);

	_objectAG.create(objectDT.size(), CV_32FC1);
	Mat objectAG = _objectAG.getMat();

	for (int i = 0; i < objectDT.rows; ++i)
		for (int j = 0; j < objectDT.cols; ++j)
			objectAG.at<float>(i, j) = objectDT.at<float>(i, j) + gray.at<uchar>(i, j);
}

//標籤切割
void LabelCut(InputArray _objectAL, InputArray _objectOpen, OutputArray _objectLC)
{
	Mat objectAL = _objectAL.getMat();
	CV_Assert(objectAL.type() == CV_8UC1);

	Mat objectOpen = _objectOpen.getMat();
	CV_Assert(objectOpen.type() == CV_8UC1);

	_objectLC.create(objectAL.size(), CV_8UC1);
	Mat objectLC = _objectLC.getMat();

	objectAL.copyTo(objectLC);

	for (int i = 0; i < objectLC.rows; ++i)
		for (int j = 0; j < objectLC.cols; ++j)
			if (objectOpen.at<uchar>(i,j) == 0)
				objectLC.at<uchar>(i, j) = 0;
}

//刪除邊界物件
void BWEdgeDelete(InputArray _object, OutputArray _objectED)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	_objectED.create(object.size(), CV_8UC1);
	Mat objectED = _objectED.getMat();

	object.copyTo(objectED);
	for (int i = 0; i < objectED.cols; ++i)
	{
		if (objectED.at<uchar>(0, i) == 255) { floodFill(objectED, Point(i, 0), 0); }
		if (objectED.at<uchar>(objectED.rows - 1, i) == 255) { floodFill(objectED, Point(i, objectED.rows - 1), 0); }
	}
	for (int i = 0; i < objectED.rows; ++i)
	{
		if (objectED.at<uchar>(i, 0) == 255) { floodFill(objectED, Point(0, i), 0); }
		if (objectED.at<uchar>(i, objectED.cols - 1) == 255) { floodFill(objectED, Point(objectED.cols - 1, i), 0); }
	}
}