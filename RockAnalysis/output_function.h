#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;
using namespace cv;

/*�M��ڵ��I*/
int findroot(int labeltable[], int label);

/*�M��s�q��*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*�N�Ƕ��Ϥ���H������*/
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage, int upperbound = 255, int lowerbound = 0);

/*�N�Ϥ���H������V�����(��J��׳��α�פ�V)*/
void DrawColorRing(InputArray _field, OutputArray _colorField);

/*�N�Ϥ���H������V�����(��J��״T�Ȥα�פ�V)*/
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorField);

/*�N�Ϥ���u�ʩԦ��åH�Ƕ������*/
void DrawGrayBar(InputArray _field, OutputArray _grayField);

/*�N���G�H�������*/
void DrawLabel(InputArray _bwImage, OutputArray _combineLabel);

/*�N���G��ܦb�m��Ϲ��W*/
void DrawEdge(InputArray _bwImage, InputArray _realImage, OutputArray _combineImage);

/*�N�ؤl�I��ܦb�쪫��W*/
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineSeed);