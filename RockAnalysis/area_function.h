#pragma once

//�h���v���ϰ�G��
//grayImage : �Ƕ��v��(8UC1)
//blurImage : �ҽk�v��(8UC1)
//divideImage : �h���v���ϰ�G��(8UC1)
void DivideArea(InputArray _grayImage, InputArray _blurImage, OutputArray _divideImage);

//����֭�
//grayImage : �Ƕ��v��(8UC1)
//bwImage : ����֭ȼv��(8UC1(BW))
//upperThreshold : ���֭�
//lowerThreshold : �C�֭�
void HysteresisThreshold(InputArray _grayImae, OutputArray _bwImage, int upperThreshold = 200, int lowerThreshold = 150);