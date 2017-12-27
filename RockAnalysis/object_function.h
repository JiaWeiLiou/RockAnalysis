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

/*H-min �ഫ*/
void HMinimaTransform(InputArray _objectDT, OutputArray _objectHMT, float H);

/*�v���κA�ǭ���*/
void Reconstruct(InputArray _marker, InputArray _mask, OutputArray _objectR);

/*�ϰ�̤p��*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue, float precision = 1.0f);

/*�̤p��*/
void MinimaDetection(InputArray _objectIM, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue);

/*�ˬd�۾F�ϰ�O�_�s�b����*/
bool CheckForAlreadyLabeledNeighbours(int x, int y, Mat &label, Point2i &outLabeledNeighbour, int &outLabel);

/*�ˬd�O�_��������*/
bool CheckIfPixelIsWatershed(int x, int y, Mat &label, Point2i &inLabeledNeighbour, int &inLabelOfNeighbour);

/*�X�i�ϰ�̤p��*/
void ExtendLocalMinimaDetection(InputArray _objectDT, OutputArray _objectEM, float H);

/*�Z���֭�*/
void DistanceCut(InputArray _objectDT, OutputArray _objectDC, float percent);

/*�W�[���аO������*/
void AddLabel(InputArray _object, InputArray _objectSeed, OutputArray _objectAL);

/*�[�`�C�ڰ�*/
void ImposeMinima(InputArray _objectDT, InputArray _objectOpen, InputArray _objectLC, OutputArray _objectIM);

/*�������ഫ*/
void WatershedTransform(InputArray _object, InputArray _objectIM, OutputArray _objectWT);

/*���X�u�P�����G����t*/
// flag = 0  -> ���鬰�զ�
// flag = 1  -> �I�����զ�
void BWCombine(InputArray _area, InputArray _line, OutputArray _object);

/*����G�ȹ�*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR);

/*�Ŭ}���*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole);

//�h�����T
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

//�|�[�Ƕ��v��
void AddGray(InputArray _objectDT, InputArray _gray, OutputArray _objectAG);

//���Ҥ���
void LabelCut(InputArray _objectAL, InputArray _objectOpen, OutputArray _objectLC);
