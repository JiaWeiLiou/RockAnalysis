#pragma once

/*��󭱪����βV�X�Ҧ�*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*����֭�*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*�h����󭱪����T*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);