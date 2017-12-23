// CombineSegmentation.cpp : �w�q�D���x���ε{�����i�J�I�C
//

#include "output_function.h"
#include "area_function.h"
#include "line_function.h"
#include "object_function.h"
#include <iostream>

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	std::cin >> infile;

	std::cout << "Please enter blur square size for Area : ";
	int blurAreaSize = 0;
	std::cin >> blurAreaSize;

	if (blurAreaSize % 2 == 0) { --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0) { --blurLineSize; }

	std::cout << "Please enter segmentational size for Seed : ";
	int seedSize = 0;
	std::cin >> seedSize;

	/*�]�w��X���W*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//�ɮ׸��|
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//�ɮצW��

	/****��¦�v���]�w****/

	/*���J��l�v��*/

	Mat image = imread(infile);			//��l�v��(8UC1 || 8UC3 )
	if (!image.data) { printf("Oh�Ano�AŪ��image���~~�I \n"); return false; }

	/*�ഫ��l�v�����Ƕ��v��*/

	Mat gray;			//�Ƕ��v��(8UC1)

	cvtColor(image, gray, CV_BGR2GRAY);

	Mat gray_R;			//��X��(8UC3)
	DrawColorBar(gray, gray_R);

	string gray_G_file = filepath + "\\" + infilename + "_0.0_GRAY(G).png";			//�ഫ��l�v�����Ƕ��v��(�Ƕ�)
	imwrite(gray_G_file, gray);
	string gray_R_file = filepath + "\\" + infilename + "_0.1_GRAY(R).png";			//�ഫ��l�v�����Ƕ��v��(����)
	imwrite(gray_R_file, gray_R);


	/****��󭱪��v���Ѩ�****/

	/*�ҽk�Ƕ��v��*/

	Mat grayBlur;			//�ҽk�Ƕ��v��(8UC1)	
	GaussianBlur(gray, grayBlur, Size(blurAreaSize, blurAreaSize), 0, 0);

	Mat grayBlur_R;			//��X��(8UC3)
	DrawColorBar(grayBlur, grayBlur_R);

	string grayBlur_G_file = filepath + "\\" + infilename + "_1.0_BLUR_I(G).png";			//�ҽk�Ƕ��v��(�Ƕ�)
	imwrite(grayBlur_G_file, grayBlur);
	string grayBlur_R_file = filepath + "\\" + infilename + "_1.1_BLUR_I(R).png";			//�ҽk�Ƕ��v��(����)
	imwrite(grayBlur_R_file, grayBlur_R);

	/*�h���ϰ�G��*/

	Mat grayDIV;			//�����Ƕ��v���ϰ�G��(8UC1)
	DivideArea(gray, grayBlur, grayDIV);

	Mat grayDIV_R;			//��X��(8UC3)
	DrawColorBar(grayDIV, grayDIV_R);

	string grayDIV_G_file = filepath + "\\" + infilename + "_2.0_DIV_I(G).png";			//�����Ƕ��v���ϰ�G��(�Ƕ�)
	imwrite(grayDIV_G_file, grayDIV);
	string grayDIV_R_file = filepath + "\\" + infilename + "_2.1_DIV_I(R).png";			//�����Ƕ��v���ϰ�G��(����)
	imwrite(grayDIV_R_file, grayDIV_R);

	/*�G�ȤƦǶ��v��*/

	Mat grayTH;			//�G�ȤƦǶ��v��(8UC1(BW))
	threshold(grayDIV, grayTH, 150, 255, THRESH_BINARY);

	string grayTH_B_file = filepath + "\\" + infilename + "_3.0_TH_I(B).png";			//�G�ȤƦǶ��v��(�G��)
	imwrite(grayTH_B_file, grayTH);

	/*��󭱪����ε��G*/

	Mat area = grayTH;			//��󭱪����ε��G(8UC1(BW))

	Mat area_L, area_I;			//��X��(8UC3�B8UC3)
	DrawLabel(area, area_L);
	DrawImage(area, image, area_I);

	string area_B_file = filepath + "\\" + infilename + "_4.0_AREA(B).png";			//��󭱪����ε��G(�G��)
	imwrite(area_B_file, area);
	string area_L_file = filepath + "\\" + infilename + "_4.1_AREA(L).png";			//��󭱪����ε��G(����)
	imwrite(area_L_file, area_L);
	string area_I_file = filepath + "\\" + infilename + "_4.2_AREA(I).png";			//��󭱪����ε��G(�|��)
	imwrite(area_I_file, area_I);


	/****���u���v���Ѩ�****/

	/*�p��v�����*/

	Mat gradx, grady;			//�����Ϋ������(16SC1)
	Differential(gray, gradx, grady);

	Mat gradf;			//��׳���(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;			//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_G, grady_G, gradm_G, gradm_R, gradd_C, gradf_C;			//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3�B8UC3)
	DrawGrayBar(gradx, gradx_G);
	DrawGrayBar(grady, grady_G);
	DrawGrayBar(gradm, gradm_G);
	DrawColorBar(gradm, gradm_R);
	DrawColorRing(gradd, gradd_C);
	DrawColorRing(gradf, gradf_C);

	string gradx_G_file = filepath + "\\" + infilename + "_5.0_GRAD_X(G).png";			//�������(�Ƕ�)
	imwrite(gradx_G_file, gradx_G);
	string grady_G_file = filepath + "\\" + infilename + "_5.1_GRAD_Y(G).png";			//�������(�Ƕ�)
	imwrite(grady_G_file, grady_G);
	string gradm_G_file = filepath + "\\" + infilename + "_5.2_GRAD_M(G).png";			//��״T��(�Ƕ�)
	imwrite(gradm_G_file, gradm_G);
	string gradm_R_file = filepath + "\\" + infilename + "_5.3_GRAD_M(R).png";			//��״T��(����)
	imwrite(gradm_R_file, gradm_R);
	string gradd_C_file = filepath + "\\" + infilename + "_5.4_GRAD_D(C).png";			//��פ�V(����)
	imwrite(gradd_C_file, gradd_C);
	string gradf_C_file = filepath + "\\" + infilename + "_5.5_GRAD_F(C).png";			//��׳���(����)
	imwrite(gradf_C_file, gradf_C);

	/*�ҽk��״T��*/

	Mat gradmBlur;			//�ҽk��״T��(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	Mat gradmBlur_R;			//��X��(8UC3)
	DrawColorBar(gradmBlur, gradmBlur_R);

	string gradmBlur_G_file = filepath + "\\" + infilename + "_6.0_BLUR_M(G).png";			//�ҽk��״T��(�Ƕ�)
	imwrite(gradmBlur_G_file, gradmBlur);
	string gradmBlur_R_file = filepath + "\\" + infilename + "_6.1_BLUR_R(G).png";			//�ҽk��״T��(����)
	imwrite(gradmBlur_R_file, gradmBlur_R);

	/*������״T�Ȱϰ�G��*/

	Mat gradmDIV;			//������״T�Ȱϰ�G��(8UC1)
	DivideLine(gradm, gradmBlur, gradmDIV);

	Mat gradmDIV_G, gradfDIV_C;			//��X��(8UC1�B8UC3)
	DrawGrayBar(gradmDIV, gradmDIV_G);
	DrawColorRing(gradmDIV, gradd, gradfDIV_C);

	string gradmDIV_file = filepath + "\\" + infilename + "_7.0_DIV_M(G).png";			//������״T�Ȱϰ�G��(�Ƕ�)
	imwrite(gradmDIV_file, gradmDIV_G);
	string gradfDIV_C_file = filepath + "\\" + infilename + "_7.1_DIV_F(C).png";		//������״T�Ȱϰ�G��(����)
	imwrite(gradfDIV_C_file, gradfDIV_C);

	/*�G�ȤƱ�״T��*/

	Mat gradmHT;			//�G�ȤƱ�״T��(8UC1(BW))
	threshold(gradmDIV, gradmHT, 1, 255, THRESH_BINARY);

	string gradmHT_B_file = filepath + "\\" + infilename + "_8.0_HT_M(B).png";			//�G�ȤƱ�״T��(�G��)
	imwrite(gradmHT_B_file, gradmHT);

	/*������νu*/

	Mat lineHC;			//������νu(8UC1(BW))
	HysteresisCut(gradmHT, area, lineHC);

	string lineHC_B_file = filepath + "\\" + infilename + "_9.0_HC_L(B).png";			//������νu(�G��)
	imwrite(lineHC_B_file, lineHC);

	/*���u�����ε��G*/

	Mat line;			//���u�����ε��G(8UC1(BW))
	BWReverse(lineHC, line);

	Mat line_L, line_I;			//��X��(8UC3�B8UC3)
	DrawLabel(line, line_L);
	DrawImage(line, image, line_I);

	string line_B_file = filepath + "\\" + infilename + "_10.0_LINE(B).png";			//���u�����ε��G(�G��)
	imwrite(line_B_file, line);
	string line_L_file = filepath + "\\" + infilename + "_10.1_LINE(L).png";			//���u�����ε��G(����)
	imwrite(line_L_file, line_L);
	string line_I_file = filepath + "\\" + infilename + "_10.2_LINE(I).png";			//���u�����ε��G(�|��)
	imwrite(line_I_file, line_I);

	/****���X���P�u���Ѩ����G****/

	/*���X���P�u*/

	Mat objectCOM;			//���X���P�u(8UC1(BW))
	BWCombine(area, line, objectCOM);

	Mat objectCOM_L, objectCOM_I;			//��X��(8UC3�B8UC3)
	DrawLabel(objectCOM, objectCOM_L);
	DrawImage(objectCOM, image, objectCOM_I);

	string  objectCOM_B_file = filepath + "\\" + infilename + "_11.0_COM_O(B).png";			//���X���P�u(�G��)
	imwrite(objectCOM_B_file, objectCOM);
	string  objectCOM_L_file = filepath + "\\" + infilename + "_11.1_COM_O(L).png";			//���X���P�u(����)
	imwrite(objectCOM_L_file, objectCOM_L);
	string  objectCOM_I_file = filepath + "\\" + infilename + "_11.2_COM_O(I).png";			//���X���P�u(�|��)
	imwrite(objectCOM_I_file, objectCOM_I);

	/*�}�B��*/

	Mat objectOpen;			//�}�B��(8UC1(BW))
	Mat element = (Mat_<uchar>(5, 5) << 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0);
	morphologyEx(objectCOM, objectOpen, MORPH_OPEN, element);

	Mat objectOpen_L, objectOpen_I;			//��X��(8UC3�B8UC3)
	DrawLabel(objectOpen, objectOpen_L);
	DrawImage(objectOpen, image, objectOpen_I);

	string  objectOpen_B_file = filepath + "\\" + infilename + "_12.0_OPEN_O(B).png";			//�}�B��(�G��)
	imwrite(objectOpen_B_file, objectOpen);
	string  objectOpen_L_file = filepath + "\\" + infilename + "_12.1_OPEN_O(L).png";			//�}�B��(����)
	imwrite(objectOpen_L_file, objectOpen_L);
	string  objectOpen_I_file = filepath + "\\" + infilename + "_12.2_OPEN_O(I).png";			//�}�B��(�|��)
	imwrite(objectOpen_I_file, objectOpen_I);

	/*�Z���ഫ*/

	Mat objectDT;		//�Z���ഫ(32FC1(BW))
	distanceTransform(objectOpen, objectDT, CV_DIST_L2, 3);

	Mat objectDT_G, objectDT_R;		//��X��(8UC1�B8UC3)
	DrawGrayBar(objectDT, objectDT_G);
	DrawColorBar(objectDT, objectDT_R);

	string objectDT_G_file = filepath + "\\" + infilename + "_13.0_DT_O(G).png";			//�Z���ഫ(�Ƕ�)
	imwrite(objectDT_G_file, objectDT_G);
	string objectDT_R_file = filepath + "\\" + infilename + "_13.1_DT_O(R).png";			//�Z���ഫ(�Ŭ�)
	imwrite(objectDT_R_file, objectDT_R);

	/*��ɪŬ}*/

	Mat objectFH;			//��ɪŬ}(8UC1(BW))
	BWFillhole(objectOpen, objectFH);

	Mat objectFH_L, objectFH_I;			//��X��(8UC3�B8UC3)
	DrawLabel(objectFH, objectFH_L);
	DrawImage(objectFH, image, objectFH_I);

	string  objectFH_B_file = filepath + "\\" + infilename + "_14.0_FH_O(B).png";			//��ɪŬ}(�G��)
	imwrite(objectFH_B_file, objectFH);
	string  objectFH_L_file = filepath + "\\" + infilename + "_14.1_FH_O(L).png";			//��ɪŬ}(����)
	imwrite(objectFH_L_file, objectFH_L);
	string  objectFH_I_file = filepath + "\\" + infilename + "_14.2_FH_O(I).png";			//��ɪŬ}(�|��)
	imwrite(objectFH_I_file, objectFH_I);

	/*�D���X�i�ϰ�̤p��*/

	Mat objectEM;		//�D���X�i�ϰ�̤p��(8UC1(BW))
	ExtendLocalMinimaDetection(objectFH, objectEM, 25);

	Mat objectEM_S;		//��X��(8UC1)
	DrawSeed(objectOpen, objectEM, objectEM_S);

	string  objectEM_B_file = filepath + "\\" + infilename + "_15.0_EM_O(B).png";			//�D���X�i�ϰ�̤p��(�G��)
	imwrite(objectEM_B_file, objectEM);
	string  objectEM_S_file = filepath + "\\" + infilename + "_15.1_EM_O(S).png";			//�D���X�i�ϰ�̤p��(�ؤl)
	imwrite(objectEM_S_file, objectEM_S);

	/*�W�[���аO������*/

	Mat objectAL;		//�W�[���аO������(8UC1(BW))
	AddLabel(objectOpen, objectEM, objectAL);

	Mat objectAL_S;		//��X��(8UC1)
	DrawSeed(objectOpen, objectAL, objectAL_S);

	string  objectAL_B_file = filepath + "\\" + infilename + "_16.0_AL_O(B).png";			//�W�[���аO������(�G��)
	imwrite(objectAL_B_file, objectAL);
	string  objectAL_S_file = filepath + "\\" + infilename + "_16.1_AL_O(S).png";			//�W�[���аO������(�ؤl)
	imwrite(objectAL_S_file, objectAL_S);

	/*�[�`�C�ڰ�*/

	Mat objectIM;		//�[�`�C�ڰ�(32FC1(BW))
	ImposeMinima(objectDT, objectAL, objectIM);

	Mat objectIM_G, objectIM_R;		//��X��(8UC1�B8UC3)
	DrawGrayBar(objectIM, objectIM_G);
	DrawColorBar(objectIM, objectIM_R);

	string objectIM_G_file = filepath + "\\" + infilename + "_17.0_IM_O(G).png";			//�[�`�C�ڰ�(�Ƕ�)
	imwrite(objectIM_G_file, objectIM_G);
	string objectIM_R_file = filepath + "\\" + infilename + "_17.1_IM_O(R).png";			//�[�`�C�ڰ�(�Ŭ�)
	imwrite(objectIM_R_file, objectIM_R);

	/*�������t��k*/

	Mat objectWT;		//�������t��k(32SC1(BW))
	WatershedTransform(objectOpen, objectIM, objectWT);

	Mat objectWT_L, objectWT_I;		//��X��(8UC3�B8UC3)
	DrawLabel(objectWT, objectWT_L);
	DrawImage(objectWT, image, objectWT_I);

	string  objectWT_B_file = filepath + "\\" + infilename + "_18.0_WT_O(B).png";			//�������t��k(�G��)
	imwrite(objectWT_B_file, objectWT);
	string  objectWT_L_file = filepath + "\\" + infilename + "_18.1_WT_O(L).png";			//�������t��k(����)
	imwrite(objectWT_L_file, objectWT_L);
	string  objectWT_I_file = filepath + "\\" + infilename + "_18.2_WT_O(I).png";			//�������t��k(�|��)
	imwrite(objectWT_I_file, objectWT_I);

	///*�Z���ഫ2*/

	//Mat objectDT2;		//�Z���ഫ2(32FC1(BW))
	//distanceTransform(objectWT, objectDT2, CV_DIST_L2, 3);

	//Mat objectDT2_G, objectDT2_R;		//��X��(8UC1�B8UC3)
	//DrawGrayBar(objectDT2, objectDT2_G);
	//DrawColorBar(objectDT2, objectDT2_R);

	//string objectDT2_G_file = filepath + "\\" + infilename + "_16.0_DT2_O(G).png";			//�Z���ഫ2(�Ƕ�)
	//imwrite(objectDT2_G_file, objectDT2_G);
	//string objectDT2_R_file = filepath + "\\" + infilename + "_16.1_DT2_O(R).png";			//�Z���ഫ2(�Ŭ�)
	//imwrite(objectDT2_R_file, objectDT2_R);

	///*��ɪŬ}2*/

	//Mat objectFH2;			//��ɪŬ}2(8UC1(BW))
	//BWFillhole(objectWT, objectFH2);

	//Mat objectFH2_L, objectFH2_I;			//��X��2(8UC3�B8UC3)
	//DrawLabel(objectFH2, objectFH2_L);
	//DrawImage(objectFH2, image, objectFH2_I);

	//string  objectFH2_B_file = filepath + "\\" + infilename + "_22.0_FH2_O(B).png";			//��ɪŬ}2(�G��)
	//imwrite(objectFH2_B_file, objectFH2);
	//string  objectFH2_L_file = filepath + "\\" + infilename + "_22.1_FH2_O(L).png";			//��ɪŬ}2(����)
	//imwrite(objectFH2_L_file, objectFH2_L);
	//string  objectFH2_I_file = filepath + "\\" + infilename + "_22.2_FH2_O(I).png";			//��ɪŬ}2(�|��)
	//imwrite(objectFH2_I_file, objectFH2_I);

	///*�D���X�i�ϰ�̤p��2*/

	//Mat objectEM2;		//�D���X�i�ϰ�̤p��2(8UC1(BW))
	//ExtendLocalMinimaDetection(objectFH2, objectEM2, 5);

	//Mat objectEM2_S;		//��X��(8UC1)
	//DrawSeed(objectWT, objectEM2, objectEM2_S);

	//string  objectEM2_B_file = filepath + "\\" + infilename + "_23.0_EM2_O(B).png";			//�D���X�i�ϰ�̤p��2(�G��)
	//imwrite(objectEM2_B_file, objectEM2);
	//string  objectEM2_S_file = filepath + "\\" + infilename + "_23.1_EM2_O(S).png";			//�D���X�i�ϰ�̤p��2(�ؤl)
	//imwrite(objectEM2_S_file, objectEM2_S);

	///*�W�[���аO������2*/

	//Mat objectAL2;		//�W�[���аO������2(8UC1(BW))
	//AddLabel(objectWT, objectEM2, objectAL2);

	//Mat objectAL2_S;		//��X��(8UC1)
	//DrawSeed(objectWT, objectAL2, objectAL2_S);

	//string  objectAL2_B_file = filepath + "\\" + infilename + "_24.0_AL2_O(B).png";			//�W�[���аO������2(�G��)
	//imwrite(objectAL2_B_file, objectAL2);
	//string  objectAL2_S_file = filepath + "\\" + infilename + "_24.1_AL2_O(S).png";			//�W�[���аO������2(�ؤl)
	//imwrite(objectAL2_S_file, objectAL2_S);

	///*�[�`�C�ڰ�2*/

	//Mat objectIM2;		//�[�`�C�ڰ�(32FC1(BW))
	//ImposeMinima(objectDT2, objectAL2, objectIM2);

	//Mat objectIM2_G, objectIM2_R;		//��X��(8UC1�B8UC3)
	//DrawGrayBar(objectIM2, objectIM2_G);
	//DrawColorBar(objectIM2, objectIM2_R);

	//string objectIM2_G_file = filepath + "\\" + infilename + "_25.0_IM2_O(G).png";			//�[�`�C�ڰ�2(�Ƕ�)
	//imwrite(objectIM2_G_file, objectIM2_G);
	//string objectIM2_R_file = filepath + "\\" + infilename + "_25.1_IM2_O(R).png";			//�[�`�C�ڰ�2(�Ŭ�)
	//imwrite(objectIM2_R_file, objectIM2_R);

	///*�������t��k2*/

	//Mat objectWT2;		//�������t��k2(32SC1(BW))
	//WatershedTransform(objectWT, objectIM2, objectWT2);

	//Mat objectWT2_L, objectWT2_I;		//��X��(8UC3�B8UC3)
	//DrawLabel(objectWT2, objectWT2_L);
	//DrawImage(objectWT2, image, objectWT2_I);

	//string  objectWT2_B_file = filepath + "\\" + infilename + "_26.0_WT2_O(B).png";			//�������t��k2(�G��)
	//imwrite(objectWT2_B_file, objectWT2);
	//string  objectWT2_L_file = filepath + "\\" + infilename + "_26.1_WT2_O(L).png";			//�������t��k2(����)
	//imwrite(objectWT2_L_file, objectWT2_L);
	//string  objectWT2_I_file = filepath + "\\" + infilename + "_26.2_WT2_O(I).png";			//�������t��k2(�|��)
	//imwrite(objectWT2_I_file, objectWT2_I);

	return 0;
}



