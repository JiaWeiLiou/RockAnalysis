#pragma once

/*�����t��*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*�p���״T�ȤΤ�V*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*���u�����βV�X�Ҧ�*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

/*�������*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC);
