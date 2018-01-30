#pragma once

// Minimum watershed segment label
#define LABEL_MINIMUM 0

// Label for the watersheds pixels (local minima)
#define LABEL_RIDGE -1

// Label for pixels that are not local minima
#define LABEL_NOLOCALMINIMUM -2

// Label for pixels that are or have been in the OrderedQueques
#define LABEL_PROCESSING -3

// Label for unprocessed pixels
#define LABEL_UNPROCESSED -4

class PixelElement
{
public:
	PixelElement(float i, int j, int k);
	float value;  ///< pixel intensity
	int x;      ///< x coordinate of pixel
	int y;      ///< y coordinate of pixel
};

class mycomparison
{
	bool reverse;
public:
	/** \brief constructor
	* \param revparam true if we want to sort from large to small*/
	mycomparison(const bool& revparam = true)
	{
		reverse = revparam;
	}

	/** \brief comparison operator: compares the values of two pixel elements
	* \param lhs first pixel element
	* \param rhs second pixel element*/
	bool operator() (PixelElement& lhs, PixelElement& rhs) const
	{
		if (reverse) return (lhs.value > rhs.value);
		else return (lhs.value < rhs.value);
	}
};

/*H-min 轉換*/
void HMinimaTransform(InputArray _objectDT, OutputArray _objectHMT, float H);

/*影像形態學重建*/
void Reconstruct(InputArray _marker, InputArray _mask, OutputArray _objectR);

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue, float precision = 1.0f);

/*最小值*/
void MinimaDetection(InputArray _objectIM, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue);

/*檢查相鄰區域是否存在標籤*/
bool CheckForAlreadyLabeledNeighbours(int x, int y, Mat &label, Point2i &outLabeledNeighbour, int &outLabel);

/*檢查是否為分水嶺*/
bool CheckIfPixelIsWatershed(int x, int y, Mat &label, Point2i &inLabeledNeighbour, int &inLabelOfNeighbour);

/*擴展區域最小值*/
void ExtendLocalMinimaDetection(InputArray _objectDT, OutputArray _objectEM, float H);

/*距離閥值*/
void DistanceCut(InputArray _objectDT, OutputArray _objectDC);

/*增加未標記之標籤*/
void AddLabel(InputArray _object, InputArray _objectSeed, OutputArray _objectAL);

/*加深低窪區*/
void ImposeMinima(InputArray _objectDT, InputArray _objectLC, OutputArray _objectIM);

/*分水嶺轉換*/
void WatershedTransform(InputArray _object, InputArray _objectIM, OutputArray _objectWT);

/*結合線與面的二值邊緣*/
// flag = 0  -> 物體為白色
// flag = 1  -> 背景為白色
void BWCombine(InputArray _area, InputArray _line, OutputArray _object);

/*反轉二值圖*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR);

/*空洞填補*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole);

//去除雜訊
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

//疊加灰階影像
void AddGray(InputArray _objectDT, InputArray _gray, OutputArray _objectAG);

//標籤切割
void LabelCut(InputArray _objectAL, InputArray _objectOpen, OutputArray _objectLC);

//刪除邊界物件
void BWEdgeDelete(InputArray _object, OutputArray _objectED);