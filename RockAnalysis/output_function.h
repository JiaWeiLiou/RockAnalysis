#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;
using namespace cv;

//�M��ڵ��I
//labeltable : ���Ҿ��
//label : ���d�ߪ����ҭ�
//return : �������ڵ��I
int findroot(int labeltable[], int label);

//�M��s�q��
//bwImage : �G�ȹϹ�(8UC1)
//labels : ���ҹϹ�(32SC1)
//nears : �۾F����(4/6/8�۾F)
//return : ���]�t0(�¦�I��)�������`�Ӽ�
int bwlabel(InputArray _bwImage, OutputArray _labels, int nears);

//�Ыئ���
//colorwheel : �������ޭ�
void makecolorwheel(vector<Scalar> &colorwheel);

//�Ыئ��
//colorbar : ������ޭ�
void makecolorbar(vector<Scalar> &colorbar);

//�N�Ƕ��Ϲ���H�Ƕ�������
//grayImage : �Ƕ��Ϲ�(8UC1/16SC1/32FC1)
//colorbarImage : �Ƕ�����Ϲ�(8UC1)
void DrawGrayBar(InputArray _grayImage, OutputArray _graybarImage);

//�N�Ƕ��Ϲ���H���Ŧ�����
//grayImage : �Ƕ��Ϲ�(8UC1/16SC1/32FC1)
//colorbarImage : ���Ŧ���Ϲ�(8UC3)
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage);

//�N��׹Ϲ���H������V���
//gradm : ��״T��(8UC1)
//gradd : ��פ�V(32FC1)
//colorringImage : ������V�Ϲ�(8UC3)
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorringImage);

//�N��׹Ϲ���H������V���
//gradf : ��׳�(16SC2)
//colorringImage : ������V�Ϲ�(8UC3)
void DrawColorRing(InputArray _gradf, OutputArray _colorringImage);

//�N�G�ȹϹ���H�������
//bwImage : �G�ȹϹ�(8UC1)
//labelImage : ���ҹϹ�(8UC3)
void DrawLabel(InputArray _bwImage, OutputArray _labelImage);

//�N�G�ȹϹ����X��l�v����H�|�X�Ϲ����
//bwImage : �G�ȹϹ�(8UC1)
//image : ��l�v��(8UC1/8UC3)
//combineImage : �|�X�Ϲ�(8UC3)
void DrawImage(InputArray _bwImage, InputArray _image, OutputArray _combineImage);

//�N�ؤl�I���X�쪫����H�|�X�Ϲ����
//object : ��l����(8UC1)
//objectSeed : �ؤl����(8UC1)
//combineImage : �|�X�Ϲ�(8UC1)
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineImae);