#pragma once

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*基於線的分割混合模式*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

/*滯後切割*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC);
