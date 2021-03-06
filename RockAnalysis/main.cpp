// CombineSegmentation.cpp : 定義主控台應用程式的進入點。
//

#include "output_function.h"
#include "area_function.h"
#include "line_function.h"
#include "object_function.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	std::cin >> infile;

	/*設定輸出文件名*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//檔案路徑
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//檔案名稱

	/****基礎影像設定****/

	/*載入原始影像*/

	time_t time = clock();

	Mat image = imread(infile);			//原始影像(8UC1 || 8UC3 )
	if (!image.data) { printf("Oh，no，讀取image錯誤~！ \n"); return false; }

	int imageMinLength = image.rows < image.cols ? image.rows : image.cols;
	
	/*轉換原始影像為灰階影像*/

	Mat gray;			//灰階影像(8UC1)

	cvtColor(image, gray, CV_BGR2GRAY);

	Mat gray_R;			//輸出用(8UC3)
	DrawColorBar(gray, gray_R);

	string gray_G_file = filepath + "\\" + infilename + "_0.0_GRAY(G).png";			//轉換原始影像為灰階影像(灰階)
	imwrite(gray_G_file, gray);
	string gray_R_file = filepath + "\\" + infilename + "_0.1_GRAY(R).png";			//轉換原始影像為灰階影像(紅藍)
	imwrite(gray_R_file, gray_R);

	/****基於面的影像萃取****/

	/*模糊灰階影像*/

	int ksize = ceil((double)imageMinLength / 10.0);
	ksize = ksize % 2 ? ksize : ksize + 1;
	double sigma = ksize / 5;

	Mat grayBlur;			//模糊灰階影像(8UC1)
	GaussianBlur(gray, grayBlur, Size(ksize, ksize), sigma, sigma);

	Mat grayBlur_R;			//輸出用(8UC3)
	DrawColorBar(grayBlur, grayBlur_R);

	string grayBlur_G_file = filepath + "\\" + infilename + "_1.0_BLUR_I(G).png";			//模糊灰階影像(灰階)
	imwrite(grayBlur_G_file, grayBlur);
	string grayBlur_R_file = filepath + "\\" + infilename + "_1.1_BLUR_I(R).png";			//模糊灰階影像(紅藍)
	imwrite(grayBlur_R_file, grayBlur_R);

	/*去除區域亮度*/

	Mat grayDIV;			//消除灰階影像區域亮度(8UC1)
	DivideArea(gray, grayBlur, grayDIV);

	Mat grayDIV_R;			//輸出用(8UC3)
	DrawColorBar(grayDIV, grayDIV_R);

	string grayDIV_G_file = filepath + "\\" + infilename + "_2.0_DIV_I(G).png";			//消除灰階影像區域亮度(灰階)
	imwrite(grayDIV_G_file, grayDIV);
	string grayDIV_R_file = filepath + "\\" + infilename + "_2.1_DIV_I(R).png";			//消除灰階影像區域亮度(紅藍)
	imwrite(grayDIV_R_file, grayDIV_R);

	/*二值化灰階影像*/



	Mat grayTH;			//二值化灰階影像(8UC1(BW))
	otsuThreshold(grayDIV, grayTH);
	//HysteresisThreshold(grayDIV, grayTH, thresholdU, thresholdD);

	string grayTH_B_file = filepath + "\\" + infilename + "_3.0_TH_I(B).png";			//二值化灰階影像(二值)
	imwrite(grayTH_B_file, grayTH);

	/*基於面的切割結果*/

	Mat area = grayTH;			//基於面的切割結果(8UC1(BW))

	Mat area_L, area_I;			//輸出用(8UC3、8UC3)
	DrawLabel(area, area_L);
	DrawImage(area, image, area_I);

	string area_B_file = filepath + "\\" + infilename + "_4.0_AREA(B).png";			//基於面的切割結果(二值)
	imwrite(area_B_file, area);
	string area_L_file = filepath + "\\" + infilename + "_4.1_AREA(L).png";			//基於面的切割結果(標籤)
	imwrite(area_L_file, area_L);
	string area_I_file = filepath + "\\" + infilename + "_4.2_AREA(I).png";			//基於面的切割結果(疊圖)
	imwrite(area_I_file, area_I);


	/****基於線的影像萃取****/

	/*計算影像梯度*/

	Mat gradx, grady;			//水平及垂直梯度(16SC1)
	Differential(gray, gradx, grady);

	Mat gradf;			//梯度場域(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;			//梯度幅值及梯度方向(8UC1、32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_G, grady_G, gradm_G, gradm_R, gradd_C, gradf_C;			//輸出用(8UC1、8UC1、8UC1、8UC3、8UC3、8UC3)
	DrawGrayBar(gradx, gradx_G);
	DrawGrayBar(grady, grady_G);
	DrawGrayBar(gradm, gradm_G);
	DrawColorBar(gradm, gradm_R);
	DrawColorRing(gradd, gradd_C);
	DrawColorRing(gradf, gradf_C);

	string gradx_G_file = filepath + "\\" + infilename + "_5.0_GRAD_X(G).png";			//水平梯度(灰階)
	imwrite(gradx_G_file, gradx_G);
	string grady_G_file = filepath + "\\" + infilename + "_5.1_GRAD_Y(G).png";			//垂直梯度(灰階)
	imwrite(grady_G_file, grady_G);
	string gradm_G_file = filepath + "\\" + infilename + "_5.2_GRAD_M(G).png";			//梯度幅值(灰階)
	imwrite(gradm_G_file, gradm_G);
	string gradm_R_file = filepath + "\\" + infilename + "_5.3_GRAD_M(R).png";			//梯度幅值(紅藍)
	imwrite(gradm_R_file, gradm_R);
	string gradd_C_file = filepath + "\\" + infilename + "_5.4_GRAD_D(C).png";			//梯度方向(色環)
	imwrite(gradd_C_file, gradd_C);
	string gradf_C_file = filepath + "\\" + infilename + "_5.5_GRAD_F(C).png";			//梯度場域(色環)
	imwrite(gradf_C_file, gradf_C);

	/*模糊梯度幅值*/

	Mat gradmBlur;			//模糊梯度幅值(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(5, 5), 1, 1);

	Mat gradmBlur_R;			//輸出用(8UC3)
	DrawColorBar(gradmBlur, gradmBlur_R);

	string gradmBlur_G_file = filepath + "\\" + infilename + "_6.0_BLUR_M(G).png";			//模糊梯度幅值(灰階)
	imwrite(gradmBlur_G_file, gradmBlur);
	string gradmBlur_R_file = filepath + "\\" + infilename + "_6.1_BLUR_R(G).png";			//模糊梯度幅值(紅藍)
	imwrite(gradmBlur_R_file, gradmBlur_R);

	/*消除梯度幅值區域亮度*/

	Mat gradmDIV;			//消除梯度幅值區域亮度(8UC1)
	DivideLine(gradm, gradmBlur, gradmDIV);

	Mat gradmDIV_G, gradmDIV_R;			//輸出用(8UC1、8UC3)
	DrawGrayBar(gradmDIV, gradmDIV_G);
	DrawColorBar(gradmDIV, gradmDIV_R);

	string gradmDIV_G_file = filepath + "\\" + infilename + "_7.0_DIV_M(G).png";		//消除梯度幅值區域亮度(灰階)
	imwrite(gradmDIV_G_file, gradmDIV_G);
	string gradmDIV_R_file = filepath + "\\" + infilename + "_7.1_DIV_M(R).png";		//消除梯度幅值區域亮度(紅藍)
	imwrite(gradmDIV_R_file, gradmDIV_R);

	/*二值化梯度幅值*/

	Mat gradmHT;			//二值化梯度幅值(8UC1(BW))
	threshold(gradmDIV, gradmHT, 1, 255, THRESH_BINARY);

	string gradmHT_B_file = filepath + "\\" + infilename + "_8.0_HT_M(B).png";			//二值化梯度幅值(二值)
	imwrite(gradmHT_B_file, gradmHT);

	/*滯後切割線*/

	Mat lineHC;			//滯後切割線(8UC1(BW))
	HysteresisCut(gradmHT, area, lineHC);

	string lineHC_B_file = filepath + "\\" + infilename + "_9.0_HC_L(B).png";			//滯後切割線(二值)
	imwrite(lineHC_B_file, lineHC);

	/*基於線的切割結果*/

	Mat line;			//基於線的切割結果(8UC1(BW))
	BWReverse(lineHC, line);

	Mat line_L, line_I;			//輸出用(8UC3、8UC3)
	DrawLabel(line, line_L);
	DrawImage(line, image, line_I);

	string line_B_file = filepath + "\\" + infilename + "_10.0_LINE(B).png";			//基於線的切割結果(二值)
	imwrite(line_B_file, line);
	string line_L_file = filepath + "\\" + infilename + "_10.1_LINE(L).png";			//基於線的切割結果(標籤)
	imwrite(line_L_file, line_L);
	string line_I_file = filepath + "\\" + infilename + "_10.2_LINE(I).png";			//基於線的切割結果(疊圖)
	imwrite(line_I_file, line_I);

	/****結合面與線的萃取結果****/

	/*結合面與線*/

	Mat objectCOM;			//結合面與線(8UC1(BW))
	BWCombine(area, line, objectCOM);

	Mat objectCOM_L, objectCOM_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectCOM, objectCOM_L);
	DrawImage(objectCOM, image, objectCOM_I);

	string  objectCOM_B_file = filepath + "\\" + infilename + "_11.0_COM_O(B).png";			//結合面與線(二值)
	imwrite(objectCOM_B_file, objectCOM);
	string  objectCOM_L_file = filepath + "\\" + infilename + "_11.1_COM_O(L).png";			//結合面與線(標籤)
	imwrite(objectCOM_L_file, objectCOM_L);
	string  objectCOM_I_file = filepath + "\\" + infilename + "_11.2_COM_O(I).png";			//結合面與線(疊圖)
	imwrite(objectCOM_I_file, objectCOM_I);

	/*開運算*/

	Mat objectOpen;			//開運算(8UC1(BW))
	//Mat elementO = (Mat_<uchar>(5,5)<< 0,1,1,1,0,\
	//								   1,1,1,1,1,\
	//								   1,1,1,1,1,\
	//								   1,1,1,1,1,\
	//								   0,1,1,1,0);
	Mat elementO = (Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	morphologyEx(objectCOM, objectOpen, MORPH_OPEN, elementO);

	Mat objectOpen_L, objectOpen_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectOpen, objectOpen_L);
	DrawImage(objectOpen, image, objectOpen_I);

	string  objectOpen_B_file = filepath + "\\" + infilename + "_12.0_OPEN_O(B).png";			//開運算(二值)
	imwrite(objectOpen_B_file, objectOpen);
	string  objectOpen_L_file = filepath + "\\" + infilename + "_12.1_OPEN_O(L).png";			//開運算(標籤)
	imwrite(objectOpen_L_file, objectOpen_L);
	string  objectOpen_I_file = filepath + "\\" + infilename + "_12.2_OPEN_O(I).png";			//開運算(疊圖)
	imwrite(objectOpen_I_file, objectOpen_I);

	/*填補空洞*/

	Mat objectFH;			//填補空洞(8UC1(BW))
	BWFillhole(objectOpen, objectFH);
	//ClearNoise(objectOpen, objectFH, noiseSize, 8, 0);

	Mat objectFH_L, objectFH_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectFH, objectFH_L);
	DrawImage(objectFH, image, objectFH_I);

	string  objectFH_B_file = filepath + "\\" + infilename + "_13.0_FH_O(B).png";			//填補空洞(二值)
	imwrite(objectFH_B_file, objectFH);
	string  objectFH_L_file = filepath + "\\" + infilename + "_13.1_FH_O(L).png";			//填補空洞(標籤)
	imwrite(objectFH_L_file, objectFH_L);
	string  objectFH_I_file = filepath + "\\" + infilename + "_13.2_FH_O(I).png";			//填補空洞(疊圖)
	imwrite(objectFH_I_file, objectFH_I);

	/*距離轉換*/

	Mat objectDT;		//距離轉換(32FC1)
	distanceTransform(objectFH, objectDT, CV_DIST_L2, 3);

	Mat objectDT_G, objectDT_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectDT, objectDT_G);
	DrawColorBar(objectDT, objectDT_R);

	string objectDT_G_file = filepath + "\\" + infilename + "_14.0_DT_O(G).png";			//距離轉換(灰階)
	imwrite(objectDT_G_file, objectDT_G);
	string objectDT_R_file = filepath + "\\" + infilename + "_14.1_DT_O(R).png";			//距離轉換(紅藍)
	imwrite(objectDT_R_file, objectDT_R);

	/*求取擴展區域最小值*/

	Mat objectEM;		//求取擴展區域最小值(8UC1(BW))
	ExtendLocalMinimaDetection(objectDT, objectEM, 3);

	Mat objectEM_S;		//輸出用(8UC1)
	DrawSeed(objectFH, objectEM, objectEM_S);

	string  objectEM_S_file = filepath + "\\" + infilename + "_15.0_EM_O(S).png";			//求取擴展區域最小值(種子)
	imwrite(objectEM_S_file, objectEM_S);

	/*距離閥值*/

	Mat objectDC;		//距離閥值(8UC1(BW))
	DistanceCut(objectDT, objectDC);
	addWeighted(objectEM, 1, objectDC, 1, 0, objectDC);

	Mat objectDC_S;		//輸出用(8UC1)
	DrawSeed(objectFH, objectDC, objectDC_S);

	string  objectDC_S_file = filepath + "\\" + infilename + "_16.0_DC_O(S).png";			//距離閥值(種子)
	imwrite(objectDC_S_file, objectDC_S);

	/*增加未標記之標籤*/

	Mat objectAL;		//增加未標記之標籤(8UC1(BW))
	AddLabel(objectFH, objectDC, objectAL);

	Mat objectAL_S;		//輸出用(8UC1)
	DrawSeed(objectFH, objectAL, objectAL_S);

	string  objectAL_S_file = filepath + "\\" + infilename + "_17.0_AL_O(S).png";			//增加未標記之標籤(種子)
	imwrite(objectAL_S_file, objectAL_S);

	/*加深低窪區*/

	Mat objectIM;		//加深低窪區(32FC1)
	ImposeMinima(objectDT, objectAL, objectIM);

	Mat objectIM_G, objectIM_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectIM, objectIM_G);
	DrawColorBar(objectIM, objectIM_R);

	string objectIM_G_file = filepath + "\\" + infilename + "_18.0_IM_O(G).png";			//加深低窪區(灰階)
	imwrite(objectIM_G_file, objectIM_G);
	string objectIM_R_file = filepath + "\\" + infilename + "_18.1_IM_O(R).png";			//加深低窪區(紅藍)
	imwrite(objectIM_R_file, objectIM_R);

	/*分水嶺演算法*/

	Mat objectWT;		//分水嶺演算法(8UC1(BW))
	WatershedTransform(objectFH, objectIM, objectWT);

	Mat objectWT_L, objectWT_I;		//輸出用(8UC3、8UC3)
	DrawLabel(objectWT, objectWT_L);
	DrawImage(objectWT, image, objectWT_I);

	string  objectWT_B_file = filepath + "\\" + infilename + "_19.0_WT_O(B).png";			//分水嶺演算法(二值)
	imwrite(objectWT_B_file, objectWT);
	string  objectWT_L_file = filepath + "\\" + infilename + "_19.1_WT_O(L).png";			//分水嶺演算法(標籤)
	imwrite(objectWT_L_file, objectWT_L);
	string  objectWT_I_file = filepath + "\\" + infilename + "_19.2_WT_O(I).png";			//分水嶺演算法(疊圖)
	imwrite(objectWT_I_file, objectWT_I);

	/*刪除邊界物件*/

	Mat objectED;		//刪除邊界物件(3SC1(BW))
	BWEdgeDelete(objectWT, objectED);

	Mat objectED_L, objectED_I;		//輸出用(8UC3、8UC3)
	DrawLabel(objectED, objectED_L);
	DrawImage(objectED, image, objectED_I);

	string  objectED_B_file = filepath + "\\" + infilename + "_20.0_ED_O(B).png";			//刪除邊界物件(二值)
	imwrite(objectED_B_file, objectED);
	string  objectED_L_file = filepath + "\\" + infilename + "_20.1_ED_O(L).png";			//刪除邊界物件(標籤)
	imwrite(objectED_L_file, objectED_L);
	string  objectED_I_file = filepath + "\\" + infilename + "_20.2_ED_O(I).png";			//刪除邊界物件(疊圖)
	imwrite(objectED_I_file, objectED_I);

	/*擬合橢圓*/

	Mat objectFE;		//擬合橢圓(8UC1)
	vector<Size2f> ellipse = DrawEllipse(objectED, objectFE);

	string  objectFE_B_file = filepath + "\\" + infilename + "_21.0_FE_O(B).png";			//擬合橢圓(二值)
	imwrite(objectFE_B_file, objectFE);

	fstream outfile;

	string outputPath = filepath + "\\" + "GrainSize.txt";

	outfile.open(outputPath, ios::out | ios::trunc);

	for (size_t i = 0; i < ellipse.size(); ++i) {
		outfile << ellipse[i].width << "\t" << ellipse[i].height << endl;
	}

	outfile.close();

	time = clock() - time;

	cout << "spend" << (float)time / CLOCKS_PER_SEC << "seconds" << endl;

	system("pause");

	return 0;
}



