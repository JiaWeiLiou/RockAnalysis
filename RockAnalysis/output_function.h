#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;
using namespace cv;

//尋找根結點
//labeltable : 標籤樹表
//label : 欲查詢的標籤值
//return : 對應的根結點
int findroot(int labeltable[], int label);

//尋找連通物
//bwImage : 二值圖像(8UC1)
//labels : 標籤圖像(32SC1)
//nears : 相鄰類型(4/6/8相鄰)
//return : 不包含0(黑色背景)的標籤總個數
int bwlabel(InputArray _bwImage, OutputArray _labels, int nears);

//創建色環
//colorwheel : 色環索引值
void makecolorwheel(vector<Scalar> &colorwheel);

//創建色條
//colorbar : 色條索引值
void makecolorbar(vector<Scalar> &colorbar);

//將灰階圖像轉以灰階色條顯示並自動拉伸
//grayImage : 灰階圖像(8UC1/16SC1/32FC1)
//colorbarImage : 灰階色條圖像(8UC1)
void DrawGrayBar(InputArray _grayImage, OutputArray _graybarImage);

//將灰階圖像轉以紅藍色條顯示並自動拉伸
//grayImage : 灰階圖像(8UC1/16SC1/32FC1)
//colorbarImage : 紅藍色條圖像(8UC3)
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage);

//將梯度圖像轉以色環方向顯示
//gradf : 梯度場(16SC2)
//colorringImage : 色環方向圖像(8UC3)
void DrawColorRing(InputArray _gradf, OutputArray _colorringImage);

//將二值圖像轉以標籤顯示
//bwImage : 二值圖像(8UC1(BW))
//labelImage : 標籤圖像(8UC3)
void DrawLabel(InputArray _bwImage, OutputArray _labelImage);

//將二值圖像結合原始影像轉以疊合圖像顯示
//bwImage : 二值圖像(8UC1(BW))
//image : 原始影像(8UC1/8UC3)
//combineImage : 疊合圖像(8UC3)
void DrawImage(InputArray _bwImage, InputArray _image, OutputArray _combineImage);

//將種子點結合原物件轉以疊合圖像顯示
//object : 原始物件(8UC1(BW))
//objectSeed : 種子物件(8UC1(BW))
//combineImage : 疊合圖像(8UC1)
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineImae);

//將二值圖像擬合橢圓並顯示
//object : 原始物件(8UC1(BW))
//ellopseImage : 橢圓圖像(8UC1(BW))
//return : 所有物件之長短軸
vector<Point2f> DrawEllipse(InputArray _object, OutputArray _ellipseImage);