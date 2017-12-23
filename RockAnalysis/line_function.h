#pragma once

//以中央差分計算灰階影像梯度
//grayImage : 灰階影像(8UC1)
//gradx : 水平梯度(8UC1)
//grady : 垂直梯度(8UC1)
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

//結合水平及垂直方向梯度為梯度場域
//gradx : 水平梯度(8UC1)
//grady : 垂直梯度(8UC1)
//gradf : 梯度場域(8UC2)
void GradientField(InputArray _gradx, InputArray _grady, OutputArray _gradf);

//以梯度場域計算梯度幅值及方向
//gradf : 梯度場域(8UC2)
//gradm : 梯度幅值(8UC1)
//gradf : 梯度方向(32FC1)
void CalculateGradient(InputArray _gradf, OutputArray _gradm, OutputArray _gradd);

//去除梯度幅值區域亮度
//gradm : 梯度幅值(8UC1)
//gradmblur : 模糊幅值(8UC1)
//gradmDivide : 去除梯度幅值區域亮度(8UC1)
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

//利用面滯後切割線
//gradm : 梯度幅值(8UC1)
//area : 面(8UC1(BW))
//line : 滯後切割線(8UC1(BW))
void HysteresisCut(InputArray _gradm, InputArray _area, OutputArray _line);
