#pragma once

//去除影像區域亮度
//grayImage : 灰階影像(8UC1)
//blurImage : 模糊影像(8UC1)
//divideImage : 去除影像區域亮度(8UC1)
void DivideArea(InputArray _grayImage, InputArray _blurImage, OutputArray _divideImage);

//滯後閥值
//grayImage : 灰階影像(8UC1)
//bwImage : 滯後閥值影像(8UC1(BW))
//upperThreshold : 高閥值
//lowerThreshold : 低閥值
void HysteresisThreshold(InputArray _grayImae, OutputArray _bwImage, int upperThreshold = 200, int lowerThreshold = 150);