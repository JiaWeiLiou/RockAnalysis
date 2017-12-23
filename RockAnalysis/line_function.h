#pragma once

//�H�����t���p��Ƕ��v�����
//grayImage : �Ƕ��v��(8UC1)
//gradx : �������(8UC1)
//grady : �������(8UC1)
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

//���X�����Ϋ�����V��׬���׳���
//gradx : �������(8UC1)
//grady : �������(8UC1)
//gradf : ��׳���(8UC2)
void GradientField(InputArray _gradx, InputArray _grady, OutputArray _gradf);

//�H��׳���p���״T�ȤΤ�V
//gradf : ��׳���(8UC2)
//gradm : ��״T��(8UC1)
//gradf : ��פ�V(32FC1)
void CalculateGradient(InputArray _gradf, OutputArray _gradm, OutputArray _gradd);

//�h����״T�Ȱϰ�G��
//gradm : ��״T��(8UC1)
//gradmblur : �ҽk�T��(8UC1)
//gradmDivide : �h����״T�Ȱϰ�G��(8UC1)
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

//�Q�έ�������νu
//gradm : ��״T��(8UC1)
//area : ��(8UC1(BW))
//line : ������νu(8UC1(BW))
void HysteresisCut(InputArray _gradm, InputArray _area, OutputArray _line);
