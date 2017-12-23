#pragma once

/*基於面的分割混合模式*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*滯後閥值*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*去除基於面的雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);