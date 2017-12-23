#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;
using namespace cv;

/*尋找根結點*/
int findroot(int labeltable[], int label);

/*尋找連通物*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*將灰階圖片轉以色條顯示*/
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage, int upperbound = 255, int lowerbound = 0);

/*將圖片轉以色環方向場顯示(輸入梯度場或梯度方向)*/
void DrawColorRing(InputArray _field, OutputArray _colorField);

/*將圖片轉以色環方向場顯示(輸入梯度幅值及梯度方向)*/
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorField);

/*將圖片轉線性拉伸並以灰階值顯示*/
void DrawGrayBar(InputArray _field, OutputArray _grayField);

/*將結果以標籤顯示*/
void DrawLabel(InputArray _bwImage, OutputArray _combineLabel);

/*將結果顯示在彩色圖像上*/
void DrawEdge(InputArray _bwImage, InputArray _realImage, OutputArray _combineImage);

/*將種子點顯示在原物件上*/
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineSeed);